import logging

from autogluon.core.models import BaggedEnsembleModel
from autogluon.tabular.models import LGBModel, RFModel, KNNModel, XGBoostModel, XTModel, TabularNeuralNetTorchModel, NNFastAiTabularModel

logger = logging.getLogger(__name__)


def rf_oob_config1(eval_metric, problem_type):
    model_cls = BaggedEnsembleModel

    model_configs = []
    for model_child in [RFModel, LGBModel]:
        model_kwargs = dict(
            hyperparameters={'use_child_oof': False},
            model_base=model_child,
            model_base_kwargs=dict(eval_metric=eval_metric, problem_type=problem_type),
        )
        fit_kwargs = dict(k_fold=8)
        model_config = dict(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )
        model_configs.append(model_config)
    return model_configs


def rf_oob_config2(eval_metric, problem_type):
    model_cls = BaggedEnsembleModel

    model_configs = []
    for model_child in [
        RFModel,
        XTModel,
        # KNNModel,
        LGBModel,
        # XGBoostModel,
        # TabularNeuralNetTorchModel,
        # NNFastAiTabularModel,
                        ]:
        model_kwargs = dict(
            # hyperparameters={'use_child_oof': False},
            model_base=model_child,
            model_base_kwargs=dict(eval_metric=eval_metric, problem_type=problem_type),
        )
        fit_kwargs = dict(k_fold=8)
        model_config = dict(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )
        model_configs.append(model_config)
    return model_configs
