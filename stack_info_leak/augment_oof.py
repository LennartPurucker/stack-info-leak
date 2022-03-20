
import logging

import numpy as np
import pandas as pd

from autogluon.core.scheduler import LocalSequentialScheduler
from autogluon.tabular.models import LGBModel, RFModel, LinearModel, KNNModel
from autogluon.core.models import BaggedEnsembleModel

from .utils import score_with_y_pred_proba

logger = logging.getLogger(__name__)


# FIXME: Only works for binary at present
def add_noise_search(y, l1_oof, problem_type, metric):
    # Train l2 model on l1_oof as only feature,
    # Add noise until l1_oof score == l2_oof score
    score_oof_kwargs = dict(y=y, problem_type=problem_type, metric=metric)

    l1_score = score_with_y_pred_proba(y_pred_proba=l1_oof, **score_oof_kwargs)

    y = y.reset_index(drop=True)
    noise_init = np.random.rand(len(l1_oof))

    def train_fn(args, reporter):
        noise_scale = args['noise_scale']
        # for noise_scale in [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        rf = RFModel(path='', name='', problem_type=problem_type, eval_metric=metric, hyperparameters={'n_estimators': 300})
        # rf = LinearModel(path='', name='', problem_type=problem_type, eval_metric=metric)
        # rf = KNNModel(path='', name='', problem_type=problem_type, eval_metric=metric, hyperparameters={'weights': 'distance'})
        # rf = LGBModel(path='', name='', problem_type=problem_type, eval_metric=metric)
        # rf = BaggedEnsembleModel(model_base=rf)
        oof_pred_proba = l1_oof.copy()
        noise = noise_init
        noise = noise * noise_scale * 2
        noise = noise - np.mean(noise)
        oof_pred_proba += noise

        X_l2 = pd.Series(oof_pred_proba, name='l1').to_frame()

        rf.fit(X=X_l2, y=y)
        l2_oof_pred_proba = rf.get_oof_pred_proba(X=X_l2, y=y)
        # l2_oof_pred_proba = rf.predict_proba(X_l2)
        l2_score = score_with_y_pred_proba(y_pred_proba=l2_oof_pred_proba, **score_oof_kwargs)
        l1_score_val_noise = score_with_y_pred_proba(y_pred_proba=oof_pred_proba, **score_oof_kwargs)
        # print(f'noise: {noise_scale}')

        score_diff_l1_noise = l1_score_val_noise - l1_score
        score_diff = l2_score - l1_score_val_noise

        # print(f'oof_noise_diff: {score_diff_l1_noise}')
        # print(f'oof           : {score_diff}')
        # print(score_diff_val)


        noise_score = -abs(score_diff) - abs(score_diff_l1_noise * 0.2)
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


def add_noise(y, l1_oof, problem_type, metric, noise_scale=0.01):
    noise_init = np.random.rand(len(l1_oof))
    oof_noise = noise_init
    oof_noise = oof_noise * noise_scale * 2
    oof_noise = oof_noise - np.mean(oof_noise)
    l1_oof_with_noise = l1_oof + oof_noise

    return l1_oof_with_noise


# TODO: CHECK CONSISTENCY AS A SEPARATE METRIC FROM L2 OOF score
#  If given 0.5 as prediction probability, what is pred proba of L2 model. Ideally L2 pred proba should increase consistently as L1 pred proba increases.
# TODO: Sample 5 different random noise inits, average results for better quality
