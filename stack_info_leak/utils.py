import copy
import logging

from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.common.utils.log_utils import set_logger_verbosity

from autogluon.core.utils import compute_weighted_metric, get_pred_from_proba

from autogluon.core.models import BaggedEnsembleModel
from autogluon.tabular.models import LGBModel, RFModel, KNNModel, XGBoostModel, XTModel
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection

logger = logging.getLogger(__name__)


def score_with_y_pred_proba(y, y_pred_proba, problem_type, metric, weights=None, quantile_levels=None):
    if metric.needs_pred:
        y_pred_proba = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=problem_type)
    return compute_weighted_metric(y, y_pred_proba, metric, weights=None, quantile_levels=None)


# TODO: Split into stages:
#  1. load / prepare data
#  2. generate L1 OOF
#  3. update L1 OOF
#  4. generate L2 OOF
#  5. calculate statistics / score
#  6. 3D graph: l1_oof, l2_oof, noise
#  7. 3D graph: score_test, score_oof, noise


def prepare_data(train_data, test_data, label, problem_type=None):
    # Separate features and labels
    X_og = train_data.drop(columns=[label])
    y_og = train_data[label]
    X_test_og = test_data.drop(columns=[label])
    y_test_og = test_data[label]

    from autogluon.core.data import LabelCleaner
    from autogluon.core.utils import infer_problem_type

    # Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
    if problem_type is None:
        problem_type = infer_problem_type(y=y_og)  # Infer problem type (or else specify directly)
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_og)
    y = label_cleaner.transform(y_og)

    feature_generator = AutoMLPipelineFeatureGenerator()
    X = feature_generator.fit_transform(X_og)
    X_test = feature_generator.transform(X_test_og)
    y_test = label_cleaner.transform(y_test_og)
    problem_type = label_cleaner.problem_type_transform
    return X, y, X_test, y_test, problem_type


def fit_l2_model(X, y, metric, problem_type):
    l2_model = BaggedEnsembleModel(
        # hyperparameters={'use_child_oof': True},
        model_base=RFModel,
        model_base_kwargs=dict(eval_metric=metric, problem_type=problem_type),
        random_state=2
    )
    l2_model.fit(X=X, y=y, k_fold=10)
    return l2_model


def template(train_data, test_data, label, metric, problem_type=None, update_l1_oof_func=None, l1_config=None, l2_config=None):
    set_logger_verbosity(0)

    update_l1_oof_func_kwargs = None
    if update_l1_oof_func is not None:
        if isinstance(update_l1_oof_func, tuple):
            update_l1_oof_func_kwargs = update_l1_oof_func[1]
            update_l1_oof_func = update_l1_oof_func[0]
        else:
            update_l1_oof_func_kwargs = dict()

    # prepare data
    X, y, X_test, y_test, problem_type = prepare_data(train_data=train_data, test_data=test_data, label=label, problem_type=problem_type)

    # fit L1 models
    l1_model_artifacts = []
    for model_config in l1_config:
        model_cls = model_config['model_cls']
        model_kwargs = model_config['model_kwargs']
        fit_kwargs = model_config['fit_kwargs']
        print(model_cls)

        l1_model = model_cls(random_state=0, **model_kwargs).fit(X=X, y=y, **fit_kwargs)

        # get L1 OOF
        l1_oof_og = l1_model.get_oof_pred_proba()
        l1_oof = copy.deepcopy(l1_oof_og)
        l1_test_pred_proba = l1_model.predict_proba(X_test)

        # update L1 OOF
        if update_l1_oof_func is not None:
            l1_oof = update_l1_oof_func(
                y=y,
                l1_oof=l1_oof,
                metric=metric,
                problem_type=problem_type,
                **update_l1_oof_func_kwargs,
            )

        l1_model_artifacts.append(
            dict(
                oof=l1_oof,
                oof_og=l1_oof_og,
                test_pred_proba=l1_test_pred_proba
            )
        )

    l1_ensemble = EnsembleSelection(ensemble_size=100, problem_type=problem_type, metric=metric)
    l1_predictions = [artifact['oof_og'] for artifact in l1_model_artifacts]
    l1_ensemble.fit(predictions=l1_predictions, labels=y)

    l1_test_pred_proba = l1_ensemble.predict_proba([artifact['test_pred_proba'] for artifact in l1_model_artifacts])
    l1_oof_og = l1_ensemble.predict_proba(l1_predictions)
    l1_oof = l1_ensemble.predict_proba([artifact['oof'] for artifact in l1_model_artifacts])

    # fit L2 model
    X_l2 = X.copy()
    X_test_l2 = X_test.copy()

    for i, artifact in enumerate(l1_model_artifacts):
        X_l2[f'__l1_pred_proba_{i}__'] = artifact['oof']
        X_test_l2[f'__l1_pred_proba_{i}__'] = artifact['test_pred_proba']

    # fit L2 models
    l2_model_artifacts = []
    for model_config in l2_config:
        model_cls = model_config['model_cls']
        model_kwargs = model_config['model_kwargs']
        fit_kwargs = model_config['fit_kwargs']
        print(model_cls)

        l2_model = model_cls(random_state=1, **model_kwargs).fit(X=X_l2, y=y, **fit_kwargs)

        # get L1 OOF
        l2_oof = l2_model.get_oof_pred_proba()
        l2_test_pred_proba = l2_model.predict_proba(X_test_l2)

        l2_model_artifacts.append(
            dict(
                oof=l2_oof,
                test_pred_proba=l2_test_pred_proba,
            )
        )

    l2_ensemble = EnsembleSelection(ensemble_size=100, problem_type=problem_type, metric=metric)
    l2_predictions = [artifact['oof'] for artifact in l2_model_artifacts]
    l2_ensemble.fit(predictions=l2_predictions, labels=y)

    l2_test_pred_proba = l2_ensemble.predict_proba([artifact['test_pred_proba'] for artifact in l2_model_artifacts])
    l2_oof = l2_ensemble.predict_proba([artifact['oof'] for artifact in l2_model_artifacts])

    # return L1 OOF, L1 OOF Modified, L2 OOF, y, metadata
    return l1_oof_og, l1_oof, l2_oof, y, y_test, l1_test_pred_proba, l2_test_pred_proba
