from stack_info_leak.augment_oof import fix_v2
from stack_info_leak.data_utils import get_dataset
from stack_info_leak.benchmark_utils import run_experiment

import logging


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    dataset = 'adult'

    train_data, test_data, metadata = get_dataset(dataset=dataset)
    label = metadata['label']
    problem_type = metadata['problem_type']
    eval_metric = metadata['eval_metric']

    sample = 5000  # Number of rows to use to train
    sample_test = 20000  # Number of rows to use to test

    if sample is not None and (sample < len(train_data)):
        train_data = train_data.sample(n=sample, random_state=0).reset_index(drop=True)

    if sample_test is not None and (sample_test < len(test_data)):
        test_data = test_data.sample(n=sample_test, random_state=0).reset_index(drop=True)

    results_list = []
    for strategy_name, strategy_func in [('Default', None), ('AddNoiseL1', fix_v2)]:
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
