import pandas as pd

from autogluon.tabular import TabularPredictor, TabularDataset

import copy
import logging

import numpy as np
import pandas as pd

from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.common.utils.log_utils import set_logger_verbosity

from autogluon.core.utils import compute_weighted_metric, get_pred_from_proba
from autogluon.core.scheduler import LocalSequentialScheduler

from autogluon.core.models import BaggedEnsembleModel
from autogluon.tabular.models import LGBModel, RFModel, KNNModel, XGBoostModel, XTModel

logger = logging.getLogger(__name__)


def score_with_y_pred_proba(y, y_pred_proba, problem_type, metric, weights=None, quantile_levels=None):
    if metric.needs_pred:
        y_pred_proba = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=problem_type)
    return compute_weighted_metric(y, y_pred_proba, metric, weights=None, quantile_levels=None)


# FIXME: Only works for binary at present
def compute_optimal_oof_noise(X, y, X_val, y_val, oof_pred_proba_init, val_pred_proba, problem_type, metric, sample_weight=None, sample_weight_val=None, quantile_levels=None):
    from autogluon.tabular.models import RFModel  # FIXME: RFModel creates circular dependence on Tabular

    score_oof_kwargs = dict(y=y, problem_type=problem_type, metric=metric, weights=sample_weight, quantile_levels=quantile_levels)
    score_val_kwargs = dict(y=y_val, problem_type=problem_type, metric=metric, weights=sample_weight_val, quantile_levels=quantile_levels)

    if not isinstance(oof_pred_proba_init, pd.DataFrame):
        oof_pred_proba_init = pd.DataFrame(oof_pred_proba_init, columns=['l1_pred_proba'])
    if not isinstance(val_pred_proba, pd.DataFrame):
        val_pred_proba = pd.DataFrame(val_pred_proba, columns=['l1_pred_proba'])

    l1_score = score_with_y_pred_proba(y_pred_proba=oof_pred_proba_init, **score_oof_kwargs)
    l1_score_val = score_with_y_pred_proba(y_pred_proba=val_pred_proba, **score_val_kwargs)

    print(l1_score)
    print(l1_score_val)

    noise_init = np.random.rand(len(oof_pred_proba_init))

    X_val_l2 = pd.concat([X_val.reset_index(drop=True), val_pred_proba], axis=1)  # FIXME: Ensure unique col names

    def train_fn(args, reporter):
        noise_scale = args['noise_scale']
        # for noise_scale in [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        rf = RFModel(path='', name='', problem_type=problem_type, eval_metric=metric, hyperparameters={'n_estimators': 50})
        # rf = KNNModel(path='', name='', problem_type=self.problem_type, eval_metric=self.eval_metric, hyperparameters={'weights': 'distance', 'n_neighbors': 200})
        oof_pred_proba = oof_pred_proba_init.copy()
        noise = noise_init
        noise = noise * noise_scale * 2
        noise = noise - np.mean(noise)
        oof_pred_proba['l1_pred_proba'] += noise

        X_l2 = pd.concat([X.reset_index(drop=True), oof_pred_proba], axis=1)  # FIXME: Ensure unique col names

        rf.fit(X=X_l2, y=y)
        l2_oof_pred_proba = rf.get_oof_pred_proba(X=X_l2, y=y)
        l2_val_pred_proba = rf.predict_proba(X=X_val_l2)
        l2_score = score_with_y_pred_proba(y_pred_proba=l2_oof_pred_proba, **score_oof_kwargs)
        l2_score_val = score_with_y_pred_proba(y_pred_proba=l2_val_pred_proba, **score_val_kwargs)

        l1_score_val_noise = score_with_y_pred_proba(y_pred_proba=oof_pred_proba, **score_oof_kwargs)
        print()
        print(f'noise: {noise_scale}')

        score_diff_l1_noise = l1_score_val_noise - l1_score
        score_diff = l2_score - l1_score
        score_diff_val = l2_score_val - l1_score_val

        print(f'oof_noise_diff: {score_diff_l1_noise}')
        print(f'oof           : {score_diff}')
        print(f'val           : {score_diff_val}')
        # print(score_diff_val)

        noise_score = score_diff_val - abs(score_diff_l1_noise * 0.1)
        print(f'noise score   : {noise_score}')

        reporter(epoch=1, accuracy=noise_score)

    from autogluon.core.space import Real
    search_space = dict(
        noise_scale=Real(0, 0.5, default=0),
    )

    scheduler = LocalSequentialScheduler(
        train_fn,
        search_space=search_space,
        resource={'num_cpus': 'all', 'num_gpus': 0},
        num_trials=30,
        reward_attr='accuracy',
        time_attr='epoch',
        checkpoint=None
    )

    scheduler.run()

    print('result:')
    print(scheduler.get_best_config())
    print(scheduler.get_best_reward())

    results = scheduler.searcher.get_results()
    for r in results:
        print(r)

    oof_noise_scale = scheduler.get_best_config()['noise_scale']

    oof_noise = noise_init
    oof_noise = oof_noise * oof_noise_scale * 2
    oof_noise = oof_noise - np.mean(oof_noise)

    return oof_noise_scale, oof_noise


def fix_v2(y, l1_oof, problem_type, metric):
    # Train l2 model on l1_oof as only feature,
    # Add noise until l1_oof score == l2_oof score
    score_oof_kwargs = dict(y=y, problem_type=problem_type, metric=metric)

    l1_score = score_with_y_pred_proba(y_pred_proba=l1_oof, **score_oof_kwargs)

    y = y.reset_index(drop=True)
    noise_init = np.random.rand(len(l1_oof))

    def train_fn(args, reporter):
        noise_scale = args['noise_scale']
        # for noise_scale in [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        rf = RFModel(path='', name='', problem_type=problem_type, eval_metric=metric, hyperparameters={'n_estimators': 50})
        oof_pred_proba = l1_oof.copy()
        noise = noise_init
        noise = noise * noise_scale * 2
        noise = noise - np.mean(noise)
        oof_pred_proba += noise

        X_l2 = pd.Series(oof_pred_proba, name='l1').to_frame()

        rf.fit(X=X_l2, y=y)
        l2_oof_pred_proba = rf.get_oof_pred_proba(X=X_l2, y=y)
        l2_score = score_with_y_pred_proba(y_pred_proba=l2_oof_pred_proba, **score_oof_kwargs)
        l1_score_val_noise = score_with_y_pred_proba(y_pred_proba=oof_pred_proba, **score_oof_kwargs)
        # print(f'noise: {noise_scale}')

        score_diff_l1_noise = l1_score_val_noise - l1_score
        score_diff = l2_score - l1_score_val_noise

        # print(f'oof_noise_diff: {score_diff_l1_noise}')
        # print(f'oof           : {score_diff}')
        # print(score_diff_val)


        noise_score = -abs(score_diff) - abs(score_diff_l1_noise * 0.1)
        print(f'noise={noise_scale},\t{round(l2_score, 3)},\t{round(l1_score_val_noise, 3)},\t{round(l1_score, 3)},\t{round(noise_score, 3)}')
        # print(f'noise score   : {noise_score}')

        reporter(epoch=1, reward=noise_score)

    from autogluon.core.space import Real
    search_space = dict(
        noise_scale=Real(0, 0.5, default=0),
    )

    scheduler = LocalSequentialScheduler(
        train_fn,
        search_space=search_space,
        resource={'num_cpus': 'all', 'num_gpus': 0},
        num_trials=100,
        time_attr='epoch',
        checkpoint=None
    )

    scheduler.run()

    # print('result:')
    # print(scheduler.get_best_config())
    # print(scheduler.get_best_reward())

    results = scheduler.searcher.get_results()
    for r in results[:5]:
        print(r)

    oof_noise_scale = scheduler.get_best_config()['noise_scale']

    oof_noise = noise_init
    oof_noise = oof_noise * oof_noise_scale * 2
    oof_noise = oof_noise - np.mean(oof_noise)

    l1_oof_with_noise = l1_oof + oof_noise

    return l1_oof_with_noise


def legacy():
    path_prefix = '../../data/airlines/'
    path_train = path_prefix + 'train_data.csv'
    path_test = path_prefix + 'test_data.csv'

    from autogluon.core.metrics import roc_auc
    eval_metric = roc_auc

    label = 'Delay'
    sample = 1000  # Number of rows to use to train
    sample_test = 20000  # Number of rows to use to test
    train_data = TabularDataset(path_train)

    if sample is not None and (sample < len(train_data)):
        train_data = train_data.sample(n=sample, random_state=0).reset_index(drop=True)

    test_data = TabularDataset(path_test)

    if sample_test is not None and (sample_test < len(test_data)):
        test_data = test_data.sample(n=sample_test, random_state=0).reset_index(drop=True)

    # Separate features and labels
    X_og = train_data.drop(columns=[label])
    y_og = train_data[label]
    X_test_og = test_data.drop(columns=[label])
    y_test_og = test_data[label]

    # X_og = X_og[['Flight']]
    # X_test_og = X_test_og[['Flight']]

    # Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
    problem_type = infer_problem_type(y=y_og)  # Infer problem type (or else specify directly)
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_og)
    y = label_cleaner.transform(y_og)

    feature_generator = AutoMLPipelineFeatureGenerator()
    X = feature_generator.fit_transform(X_og)

    X_test = feature_generator.transform(X_test_og)
    y_test = label_cleaner.transform(y_test_og)

    problem_type = label_cleaner.problem_type_transform

    from autogluon.core.models import BaggedEnsembleModel
    from autogluon.tabular.models import LGBModel, RFModel, KNNModel
    model_l1 = BaggedEnsembleModel(
        hyperparameters={'use_child_oof': True},
        # model_base=RFModel(hyperparameters={'n_estimators': 5000}, eval_metric=eval_metric, problem_type=problem_type),

        model_base=KNNModel(eval_metric=eval_metric, problem_type=problem_type),
        random_state=1
    )
    model_l1.fit(X=X, y=y, k_fold=10)  # Fit custom model
    oof_pred_proba_l1 = model_l1.get_oof_pred_proba()
    test_pred_proba_l1 = model_l1.predict_proba(X_test)

    oof_score_l1 = roc_auc(y, oof_pred_proba_l1)
    test_score_l1 = roc_auc(y_test, test_pred_proba_l1)

    model_l2 = BaggedEnsembleModel(
        hyperparameters={'use_child_oof': True},
        model_base=RFModel(hyperparameters={'n_estimators': 5000}, eval_metric=eval_metric, problem_type=problem_type),
        random_state=2
    )
    # X_l2 = pd.DataFrame(oof_pred_proba_l1, columns=['l1_pred_proba'])
    # X_test_l2 = pd.DataFrame(test_pred_proba_l1, columns=['l1_pred_proba'])

    # oof_noise_scale, oof_noise = compute_optimal_oof_noise(
    #     X=X,
    #     y=y,
    #     X_val=X_test,
    #     y_val=y_test,
    #     oof_pred_proba_init=pd.Series(oof_pred_proba_l1),
    #     val_pred_proba=pd.Series(test_pred_proba_l1),
    #     metric=eval_metric,
    #     problem_type=problem_type
    # )

    oof_noise_scale, oof_noise = fix_v2(
        y=y,
        l1_oof=oof_pred_proba_l1,
        metric=eval_metric,
        problem_type=problem_type
    )

    # oof_pred_proba_l1 += oof_noise

    X_l2 = X.copy()
    X_l2['l1_pred_proba'] = oof_pred_proba_l1
    X_test_l2 = X_test.copy()
    X_test_l2['l1_pred_proba'] = test_pred_proba_l1

    model_l2.fit(X=X_l2, y=y, k_fold=10)
    oof_pred_proba_l2 = model_l2.get_oof_pred_proba()
    test_pred_proba_l2 = model_l2.predict_proba(X_test_l2)
    oof_score_l2 = roc_auc(y, oof_pred_proba_l2)
    test_score_l2 = roc_auc(y_test, test_pred_proba_l2)

    df_out = pd.concat([pd.Series(oof_pred_proba_l1, name='l1'), pd.Series(oof_pred_proba_l2, name='l2'), y], axis=1)

    print('Label = 0')
    print(df_out[df_out[label] == 0].mean())

    print('Label = 1')
    print(df_out[df_out[label] == 1].mean())


def graph_oof(oof_l1, oof_l2, y_true):
    from matplotlib import pyplot as plt
    x = oof_l1
    y = oof_l2 - oof_l1

    plt.scatter(x, y, c=y_true, alpha=0.5)
    plt.show()


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


def fit_l1_model(X, y, metric, problem_type):
    model_base = RFModel(eval_metric=metric, problem_type=problem_type)
    # model_base = XTModel(eval_metric=metric, problem_type=problem_type)
    # model_base = KNNModel(eval_metric=metric, problem_type=problem_type)
    l1_model = BaggedEnsembleModel(
        # hyperparameters={'use_child_oof': True},
        model_base=model_base,
        random_state=1
    )
    l1_model.fit(X=X, y=y, k_fold=10)
    return l1_model


def fit_l2_model(X, y, metric, problem_type):
    # model_base = LGBModel(eval_metric=metric, problem_type=problem_type)
    model_base = RFModel(eval_metric=metric, problem_type=problem_type)
    # model_base = XGBoostModel(eval_metric=metric, problem_type=problem_type)
    l2_model = BaggedEnsembleModel(
        hyperparameters={'use_child_oof': True},
        model_base=model_base,
        random_state=2
    )
    l2_model.fit(X=X, y=y, k_fold=10)
    return l2_model


def template(train_data, test_data, label, metric, problem_type=None, update_l1_oof_func=None, update_l1_oof_func_kwargs=None):
    set_logger_verbosity(0)
    # prepare data
    X, y, X_test, y_test, problem_type = prepare_data(train_data=train_data, test_data=test_data, label=label, problem_type=problem_type)

    # fit L1 model
    l1_model = fit_l1_model(X=X, y=y, metric=metric, problem_type=problem_type)

    # get L1 OOF
    l1_oof_og = l1_model.get_oof_pred_proba()
    l1_oof = copy.deepcopy(l1_oof_og)
    l1_test_pred_proba = l1_model.predict_proba(X_test)

    # update L1 OOF
    if update_l1_oof_func is not None:
        if update_l1_oof_func_kwargs is None:
            update_l1_oof_func_kwargs = dict()
        l1_oof = update_l1_oof_func(
            y=y,
            l1_oof=l1_oof,
            metric=metric,
            problem_type=problem_type,
            **update_l1_oof_func_kwargs,
        )

    # fit L2 model
    X_l2 = X.copy()
    X_l2['__l1_pred_proba__'] = l1_oof
    X_test_l2 = X_test.copy()
    X_test_l2['__l1_pred_proba__'] = l1_test_pred_proba
    l2_model = fit_l2_model(X=X_l2, y=y, metric=metric, problem_type=problem_type)

    # get L2 OOF
    l2_oof = l2_model.get_oof_pred_proba()
    l2_test_pred_proba = l2_model.predict_proba(X_test_l2)

    # return L1 OOF, L1 OOF Modified, L2 OOF, y, metadata
    return l1_oof_og, l1_oof, l2_oof, y, y_test, l1_test_pred_proba, l2_test_pred_proba


if __name__ == '__main__':
    label = 'Delay'
    # import sklearn.datasets as dt
    # rand_state = 0
    # len_synthetic = 20000
    # X, y = dt.make_classification(n_samples=len_synthetic,
    #                               n_features=1,
    #                               flip_y=0.8,
    #                               n_redundant=0,
    #                               n_repeated=0,
    #                               n_informative=1,
    #                               n_clusters_per_class=1,
    #                               random_state=rand_state)
    # data = pd.DataFrame(X)
    # data[label] = y
    #
    # train_data = data[:10000]
    # test_data = data[10000:]
    # from autogluon.core.metrics import root_mean_squared_error
    # eval_metric = root_mean_squared_error
    # synthetic_data = pd.DataFrame([
    #     np.random.rand(len_synthetic)
    # ]
    #
    # )




    path_prefix = '../../data/airlines/'
    path_train = path_prefix + 'train_data.csv'
    path_test = path_prefix + 'test_data.csv'
    #
    from autogluon.core.metrics import roc_auc
    eval_metric = roc_auc
    #
    # label = 'Delay'
    sample = 5000  # Number of rows to use to train
    sample_test = 20000  # Number of rows to use to test
    train_data = TabularDataset(path_train)
    #
    if sample is not None and (sample < len(train_data)):
        train_data = train_data.sample(n=sample, random_state=0).reset_index(drop=True)
    #
    test_data = TabularDataset(path_test)

    if sample_test is not None and (sample_test < len(test_data)):
        test_data = test_data.sample(n=sample_test, random_state=0).reset_index(drop=True)

    for strategy_name, strategy_func in [('Default', None), ('AddNoiseL1', fix_v2)]:
        print(f'Strategy: {strategy_name}')
        l1_oof_og, l1_oof, l2_oof, y, y_test, l1_test_pred_proba, l2_test_pred_proba = template(
            train_data=train_data,
            test_data=test_data,
            label=label,
            metric=eval_metric,
            problem_type=None,
            update_l1_oof_func=strategy_func,
            update_l1_oof_func_kwargs=None,
        )

        l1_score_oof_og = eval_metric(y, l1_oof_og)
        l1_score_oof = eval_metric(y, l1_oof)
        l1_score_test = eval_metric(y_test, l1_test_pred_proba)

        l2_score_oof = eval_metric(y, l2_oof)
        l2_score_test = eval_metric(y_test, l2_test_pred_proba)

        # df_out = pd.concat([pd.Series(l1_oof_og, name='l1'), pd.Series(l2_oof, name='l2'), y], axis=1)
        # print('Label = 0')
        # print(df_out[df_out[label] == 0].mean())
        # print('Label = 1')
        # print(df_out[df_out[label] == 1].mean())

        print(f'l1_score_oof_og: {l1_score_oof_og}')
        print(f'l1_score_oof:    {l1_score_oof}')
        print(f'l1_score_test:   {l1_score_test}')
        print(f'l2_score_oof:    {l2_score_oof}')
        print(f'l2_score_test:   {l2_score_test}')
