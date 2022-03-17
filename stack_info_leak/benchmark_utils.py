
import logging
from .utils import template

logger = logging.getLogger(__name__)


def run_experiment(
        train_data,
        test_data,
        label,
        eval_metric,
        strategy_name,
        strategy_func):
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

    result_dict = dict(
        strategy_name=strategy_name,
        l1_score_oof_og=l1_score_oof_og,
        l1_score_oof=l1_score_oof,
        l1_score_test=l1_score_test,
        l2_score_oof=l2_score_oof,
        l2_score_test=l2_score_test,
    )
    return result_dict
