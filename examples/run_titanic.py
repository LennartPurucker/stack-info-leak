import logging

import pandas as pd

from stack_info_leak.augment_oof import add_noise_search, add_noise
from stack_info_leak.data_utils import get_dataset
from stack_info_leak.benchmark_utils import run_experiment

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    dataset = 'titanic'

    train_data, test_data, metadata = get_dataset(dataset=dataset)
    label = metadata['label']
    problem_type = metadata['problem_type']
    eval_metric = metadata['eval_metric']

    from stack_info_leak.model_utils import rf_oob_config2
    l1_config = rf_oob_config2(eval_metric, problem_type, random_state=1)
    l2_config = rf_oob_config2(eval_metric, problem_type, random_state=2)

    results_list = []
    for strategy_name, strategy_func in [
        ('Default', None),
        # ('AddNoise', add_noise_search),
        ('AddNoise_0.01', (add_noise, {'noise_scale': 0.01})),
        ('AddNoise_0.1', (add_noise, {'noise_scale': 0.1})),
        ('AddNoise_0.2', (add_noise, {'noise_scale': 0.2})),
        ('AddNoise_0.5', (add_noise, {'noise_scale': 0.5})),
        ('AddNoise_1.0', (add_noise, {'noise_scale': 1})),
    ]:
        print(f'Strategy: {strategy_name}')
        result_dict = run_experiment(
            train_data=train_data,
            test_data=test_data,
            label=label,
            eval_metric=eval_metric,
            strategy_name=strategy_name,
            strategy_func=strategy_func,
            l1_config=l1_config,
            l2_config=l2_config,
            problem_type=problem_type,
        )
        results_list.append(result_dict)

    results_df = pd.DataFrame(results_list)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)