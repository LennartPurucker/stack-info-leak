from autogluon.tabular import TabularDataset
from stack_info_leak.utils import template
from stack_info_leak.augment_oof import fix_v2

import logging


logger = logging.getLogger(__name__)


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
