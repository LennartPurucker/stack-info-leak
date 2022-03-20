
import logging
from .utils import template, score_with_y_pred_proba

logger = logging.getLogger(__name__)


def run_experiment(
        train_data,
        test_data,
        label,
        eval_metric,
        strategy_name,
        strategy_func,
        l1_config,
        l2_config,
        problem_type=None,):
    l1_oof_og, l1_oof, l2_oof, y, y_test, l1_test_pred_proba, l2_test_pred_proba = template(
        train_data=train_data,
        test_data=test_data,
        label=label,
        metric=eval_metric,
        problem_type=problem_type,
        update_l1_oof_func=strategy_func,
        l1_config=l1_config,
        l2_config=l2_config,
    )

    l1_score_oof_og = score_with_y_pred_proba(y, l1_oof_og, metric=eval_metric, problem_type=problem_type)
    l1_score_oof = score_with_y_pred_proba(y, l1_oof, metric=eval_metric, problem_type=problem_type)
    l1_score_test = score_with_y_pred_proba(y_test, l1_test_pred_proba, metric=eval_metric, problem_type=problem_type)
    l2_score_oof = score_with_y_pred_proba(y, l2_oof, metric=eval_metric, problem_type=problem_type)
    l2_score_test = score_with_y_pred_proba(y_test, l2_test_pred_proba, metric=eval_metric, problem_type=problem_type)
    if l1_score_oof_og >= l2_score_oof:
        score_test = l1_score_test
    else:
        score_test = l2_score_test

    result_dict = dict(
        strategy_name=strategy_name,
        score_test=score_test,
        l1_score_test=l1_score_test,
        l2_score_test=l2_score_test,
        l1_score_oof_og=l1_score_oof_og,
        l1_score_oof=l1_score_oof,
        l2_score_oof=l2_score_oof,
    )
    return result_dict
