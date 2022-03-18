import logging

from autogluon.tabular import TabularDataset
from autogluon.common.savers import save_pd
from autogluon.core.metrics import roc_auc
from autogluon.core.utils import generate_train_test_split_combined_X_y

logger = logging.getLogger(__name__)


DATASETS = dict(
    adult=dict(
        format='s3',
        train_path='s3://autogluon/datasets/AdultIncomeBinaryClassification/train_data.csv',
        test_path='s3://autogluon/datasets/AdultIncomeBinaryClassification/test_data.csv',
        cache=True,
        eval_metric=roc_auc,
        problem_type='binary',
        label='class',
    ),
    santander_customer_satisfaction=dict(
        format='s3',
        train_path='s3://autogluon/datasets/santander-customer-satisfaction/train.csv',
        test_size=0.3,
        cache=True,
        eval_metric=roc_auc,
        problem_type='binary',
        label='TARGET',
        drop_columns=['ID'],
    ),
)


def load_dataset_and_cache(path, path_cache=None, save_cache=True):
    is_loaded = False
    data = None
    if path_cache is not None:
        try:
            data = TabularDataset(path_cache)
            is_loaded = True
        except:
            pass
    if not is_loaded:
        data = TabularDataset(path)
        if path_cache is not None and save_cache:
            save_pd.save(df=data, path=path_cache)
    return data


def get_dataset(dataset: str):
    if dataset not in DATASETS:
        raise AssertionError(f'{dataset} not in valid datasets: {list(DATASETS.keys())}')
    d = DATASETS[dataset]
    if d['format'] == 's3':
        train_data = load_dataset_and_cache(path=d['train_path'], path_cache=f'datasets/{dataset}/train_data.csv', save_cache=d['cache'])
        if 'test_data' in d:
            test_data = load_dataset_and_cache(path=d['test_path'], path_cache=f'datasets/{dataset}/test_data.csv', save_cache=d['cache'])
        else:
            train_data, test_data = generate_train_test_split_combined_X_y(data=train_data,
                                                                           label=d['label'],
                                                                           problem_type=d['problem_type'],
                                                                           test_size=d['test_size'],
                                                                           random_state=0)
        if 'drop_columns' in d:
            train_data = train_data.drop(columns=d['drop_columns'])
            test_data = test_data.drop(columns=d['drop_columns'])
        return train_data, test_data, d.copy()
    else:
        raise AssertionError(f'Invalid format: {d["format"]}')

