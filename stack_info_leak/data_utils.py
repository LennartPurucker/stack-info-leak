import logging

from autogluon.tabular import TabularDataset
from autogluon.common.savers import save_pd
from autogluon.core.metrics import roc_auc

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
    )
)


def get_dataset(dataset: str):
    if dataset not in DATASETS:
        raise AssertionError(f'{dataset} not in valid datasets: {list(DATASETS.keys())}')
    d = DATASETS[dataset]
    if d['format'] == 's3':
        is_loaded = False
        if d['cache'] is True:
            try:
                train_data = TabularDataset(f'datasets/{dataset}/train_data.csv')
                test_data = TabularDataset(f'datasets/{dataset}/test_data.csv')
                is_loaded = True
            except:
                pass
        if not is_loaded:
            train_data = TabularDataset(d['train_path'])
            test_data = TabularDataset(d['test_path'])

            save_pd.save(df=train_data, path=f'datasets/{dataset}/train_data.csv')
            save_pd.save(df=test_data, path=f'datasets/{dataset}/test_data.csv')
        return train_data, test_data, d.copy()
    else:
        raise AssertionError(f'Invalid format: {d["format"]}')

