import logging

import numpy as np
import pandas as pd
import copy
from fairlearn.preprocessing import CorrelationRemover
from autogluon.core.scheduler import LocalSequentialScheduler
from autogluon.tabular.models import LGBModel, RFModel, LinearModel, KNNModel
from autogluon.core.models import BaggedEnsembleModel

from .utils import score_with_y_pred_proba
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import softmax
logger = logging.getLogger(__name__)


# FIXME: Only works for binary at present
def add_noise_search(y, l1_oof, l1_test_pred_proba, problem_type, metric, l1_repro_pred=None, **kwargs):
    # Train l2 model on l1_oof as only feature,
    # Add noise until l1_oof score == l2_oof score
    score_oof_kwargs = dict(y=y, problem_type=problem_type, metric=metric)

    l1_score = score_with_y_pred_proba(y_pred_proba=l1_oof, **score_oof_kwargs)

    y = y.reset_index(drop=True)
    noise_init = np.random.rand(len(l1_oof))

    def train_fn(args, reporter):
        noise_scale = args['noise_scale']
        # for noise_scale in [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        rf = RFModel(path='', name='', problem_type=problem_type, eval_metric=metric,
                     hyperparameters={'n_estimators': 50})
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
        repro_X_l2 = pd.Series(l1_repro_pred, name="l1").to_frame()

        rf.fit(X=X_l2, y=y)
        l2_oof_pred_proba = rf.get_oof_pred_proba(X=X_l2, y=y)

        # repro_X_l2_pred_proba = rf.get_oof_pred_proba(X=repro_X_l2, y=y)
        # repro_score = score_with_y_pred_proba(y_pred_proba=repro_X_l2_pred_proba, **score_oof_kwargs)

        l2_score = score_with_y_pred_proba(y_pred_proba=l2_oof_pred_proba, **score_oof_kwargs)
        l1_score_val_noise = score_with_y_pred_proba(y_pred_proba=oof_pred_proba, **score_oof_kwargs)
        # print(f'noise: {noise_scale}')

        score_diff_l1_noise = l1_score_val_noise - l1_score
        score_diff = l2_score - l1_score_val_noise

        # print(f'oof_noise_diff: {score_diff_l1_noise}')
        # print(f'oof           : {score_diff}')
        # print(score_diff_val)

        noise_score = -abs(score_diff) - abs(score_diff_l1_noise * 0.2)
        print(
            f'noise={noise_scale},\t{round(l2_score, 3)},\t{round(l1_score_val_noise, 3)},\t{round(l1_score, 3)},\t{round(noise_score, 3)}')
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
        num_trials=20,
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

    return l1_oof_with_noise, l1_test_pred_proba


def add_noise(y, l1_oof,l1_test_pred_proba, problem_type, metric, noise_scale=0.01, **kwargs):
    noise_init = np.random.rand(len(l1_oof))
    oof_noise = noise_init
    oof_noise = oof_noise * noise_scale * 2
    oof_noise = oof_noise - np.mean(oof_noise)
    l1_oof_with_noise = l1_oof + oof_noise

    return l1_oof_with_noise, l1_test_pred_proba


def add_noise_via_swap(y, l1_oof, l1_test_pred_proba, problem_type, metric, strength=0.1, **kwargs):
    a = pd.Series(l1_oof)
    b = pd.concat([a, y.reset_index(drop=True)], axis=1)
    c = b.sort_values(by=0)
    c['_is_swapped'] = 0
    d = c.copy()

    len_c = len(c)
    swap_max = np.ceil(len_c * strength)
    rand_vals = np.random.randint(1, swap_max, len_c)

    for j in range(len_c):
        for i, pos in [(j, True), (j, False)]:
            if pos:
                i_swap = i + rand_vals[i]
            else:
                i = -i - 1
                i_swap = i - rand_vals[i]

            if i_swap >= len_c or i_swap < -len_c:
                continue
            if d.iloc[i, 2] != 0:
                continue
            if d.iloc[i_swap, 2] != 0:
                continue
            v1 = d.iloc[i, 0]
            v2 = d.iloc[i_swap, 0]
            d.iloc[i, 0] = v2
            d.iloc[i_swap, 0] = v1
            d.iloc[i, 2] = 1
            d.iloc[i_swap, 2] = 1

    out = d[0].sort_index().to_numpy()
    return out, l1_test_pred_proba


def add_noise_via_dropout(y, l1_oof,l1_test_pred_proba, problem_type, metric, dropout=0.1, **kwargs):
    m = np.mean(l1_oof)
    a = pd.Series(l1_oof)
    b = pd.concat([a, y.reset_index(drop=True)], axis=1)
    b['_is_dropout'] = 0
    d = b.copy()
    len_c = len(b)
    rand_vals = np.random.random(len_c)

    for i in range(len_c):
        if rand_vals[i] <= dropout:
            d.iloc[i, 0] = m
            d.iloc[i, 2] = 1

    out = d[0].to_numpy()
    return out,l1_test_pred_proba


def eval_oof_fairness(l2_X, fold_indicator):
    f_X_train, f_X_test, f_y_train, f_y_test = train_test_split(l2_X, fold_indicator, test_size=0.33,
                                                                random_state=42, stratify=fold_indicator)
    fairness_model = RandomForestClassifier(random_state=1).fit(f_X_train, f_y_train)
    y_pred = fairness_model.predict(f_X_test)
    fairness_score = balanced_accuracy_score(f_y_test, y_pred)
    return fairness_score

def make_oof_fair(y, l1_oof,l1_test_pred_proba, problem_type, metric, _X=None, fold_indicator=None, **kwargs):

    # TODO Ideas
    #   - real fairness methods and other stuff (need to read up on it)
    #   - optimize for minimal fairness score but make sure to avoid change proba data acutal data
    #   - In general, do apply fixes per fold or on average?

    # Try 1:
    # - standard scaler per fold? I do not think that this will fix it.
    #   underlying distribution should not change really.
    # new_oof = l1_oof.copy()
    # max_fold = np.max(fold_indicator)
    # for i in range(max_fold):
    #     fold_instances = fold_indicator == i
    #     # new_oof[fold_instances] = MinMaxScaler().fit_transform(StandardScaler().fit_transform(new_oof[fold_instances].reshape(-1,1))).flatten()
    #
    #     if i == 0:
    #         iser = IsotonicRegression(y_min=0, y_max=1, increasing="auto",out_of_bounds="clip")
    #         new_oof[fold_instances] = iser.fit_transform(new_oof[fold_instances], y[fold_instances])
    #     else:
    #         new_oof[fold_instances] = iser.transform(new_oof[fold_instances])
    # out = new_oof

    new_oof = copy.deepcopy(l1_oof)
    new_test = copy.deepcopy(l1_test_pred_proba)
    # t = pd.DataFrame()
    # t["f"] = fold_indicator
    # t["oof"] = new_oof
    # x = CorrelationRemover(sensitive_feature_ids=["f"],alpha=1).fit_transform(t)

    # Low association score but high overfitting, why the fuck?
    # mask = new_oof >=0.5
    # new_oof[mask] = 1.0
    # new_oof[~mask] = 0.0
    # mask = new_test >= 0.5
    # new_test[mask] = 1.0
    # new_test[~mask] = 0.0

    out = new_oof
    out2 = new_test
    # l2_X = _X.copy()
    # l2_X["oof"] = l1_oof
    # fairness = eval_oof_fairness(l2_X, fold_indicator)

    # Round, reduces but not removes distrubtional differences
    #   e.g. 0.7/0.3 vs. 0.8/0.2 is not removed.
    #   Moreover. we lose potentially very useful information.
    #   Overfits heavily because completely different data in the end
    #   -> how is that possible?!!!
    #   -> predicting fold association becomes harder but overfit higher, wtf is this?
    #   -> need to round test data as well perhaps,
    #out = np.around(l1_oof, 1)

    # from scipy.special import softmax
    # Softmax idea
    # out = MinMaxScaler().fit_transform(softmax(l1_oof).reshape(-1,1)).flatten()
    # this just scales again, does not change the distributions...
    return out, out2

# TODO: CHECK CONSISTENCY AS A SEPARATE METRIC FROM L2 OOF score
#  If given 0.5 as prediction probability, what is pred proba of L2 model. Ideally L2 pred proba should increase consistently as L1 pred proba increases.
# TODO: Sample 5 different random noise inits, average results for better quality


#   strategy_name  score_test  l2_overfit  l1_score_test  l2_score_test  l1_score_oof  l2_score_oof  l1_score_oof_aug
# -1  (variable)    AddNoise    0.531539    0.001864       0.531539       0.522807      0.525939      0.524671          0.494846
# 1       Default    0.515847    0.047132       0.532794       0.515847      0.525939      0.562979          0.525939
# 2      MakeFairFoldCalibration    0.522708    0.010853       0.531539       0.522708      0.525939       0.53356          0.500649