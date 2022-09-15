import logging

import numpy as np
import pandas as pd

from autogluon.tabular import TabularDataset
from autogluon.common.savers import save_pd
from autogluon.core.metrics import accuracy, roc_auc
# from autogluon.core.utils import generate_train_test_split_combined
from .split_utils import generate_train_test_split_combined

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
    airlines=dict(
        format='s3',
        train_path='s3://autogluon/datasets/leakage/airlines/train_data.csv',
        test_path='s3://autogluon/datasets/leakage/airlines/test_data.csv',
        cache=True,
        eval_metric=roc_auc,
        problem_type='binary',
        label='Delay',
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
    titanic=dict(
        format='s3',
        train_path='s3://autogluon/datasets/titanic/train.csv',
        test_size=0.5,
        cache=True,
        eval_metric=accuracy,
        problem_type='binary',
        label='Survived',
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


def get_dataset(dataset: str, sample=None, sample_test=None):
    train_data, test_data, metadata = _get_dataset(dataset=dataset)
    if sample is not None and (sample < len(train_data)):
        train_data = train_data.sample(n=sample, random_state=0).reset_index(drop=True)
    if sample_test is not None and (sample_test < len(test_data)):
        test_data = test_data.sample(n=sample_test, random_state=0).reset_index(drop=True)
    return train_data, test_data, metadata

def _get_dataset(dataset: str):
    if dataset not in DATASETS:
        raise AssertionError(f'{dataset} not in valid datasets: {list(DATASETS.keys())}')
    d = DATASETS[dataset]
    if d['format'] == 's3':
        train_data = load_dataset_and_cache(path=d['train_path'], path_cache=f'datasets/{dataset}/train_data.csv', save_cache=d['cache'])
        if 'test_path' in d:
            test_data = load_dataset_and_cache(path=d['test_path'], path_cache=f'datasets/{dataset}/test_data.csv', save_cache=d['cache'])
        else:
            train_data, test_data = generate_train_test_split_combined(data=train_data,
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

