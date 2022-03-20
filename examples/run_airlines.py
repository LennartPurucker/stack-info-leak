from autogluon.tabular import TabularDataset
from stack_info_leak.augment_oof import add_noise_search
from stack_info_leak.benchmark_utils import run_experiment

import logging


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    label = 'Delay'

    path_prefix = '../../data/airlines/'
    path_train = path_prefix + 'train_data.csv'
    path_test = path_prefix + 'test_data.csv'
    from autogluon.core.metrics import roc_auc
    eval_metric = roc_auc
    problem_type = 'binary'

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

    results_list = []
    for strategy_name, strategy_func in [('Default', None), ('AddNoiseL1', add_noise_search)]:
        print(f'Strategy: {strategy_name}')
        result_dict = run_experiment(
            train_data=train_data,
            test_data=test_data,
            label=label,
            eval_metric=eval_metric,
            strategy_name=strategy_name,
            strategy_func=strategy_func,
            problem_type=problem_type,
        )
        results_list.append(result_dict)

    for result_dict in results_list:
        for key in result_dict:
            print(f'{key}:\t{result_dict[key]}')
