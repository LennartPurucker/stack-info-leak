from stack_info_leak.augment_oof import add_noise_search
from stack_info_leak.data_utils import get_dataset
from stack_info_leak.benchmark_utils import run_experiment
import pandas as pd
import logging


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    dataset = 'adult'

    train_data, test_data, metadata = get_dataset(dataset=dataset)
    label = metadata['label']
    problem_type = metadata['problem_type']
    eval_metric = metadata['eval_metric']

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

    results_df = pd.DataFrame(results_list)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)