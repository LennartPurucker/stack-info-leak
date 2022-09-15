from pandas import DataFrame, Series

from autogluon.core.utils import generate_train_test_split


def extract_label(data: DataFrame, label: str) -> (DataFrame, Series):
    """
    Extract the label column from a dataset and return X, y.

    Parameters
    ----------
    data : DataFrame
        The data containing features and the label column.
    label : str
        The label column name.

    Returns
    -------
    X, y : (DataFrame, Series)
        X is the data with the label column dropped.
        y is the label column as a pd.Series.
    """
    if label not in list(data.columns):
        raise ValueError(f"Provided DataFrame does not contain label column: {label}")
    y = data[label].copy()
    X = data.drop(label, axis=1)
    return X, y


def generate_train_test_split_combined(data: DataFrame,
                                       label: str,
                                       problem_type: str,
                                       test_size: float = 0.1,
                                       random_state: int = 0,
                                       min_cls_count_train: int = 1) -> (DataFrame, DataFrame):
    """
    Generate a train test split from a DataFrame that contains the label column.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the features plus the label column to split into train and test sets.
    label : str
        The label column name.
        Used for stratification and to ensure all classes in multiclass classification are preserved in train data.
    problem_type : str
        The problem_type the label is used for. Determines if stratification is used.
        Options: ["binary", "multiclass", "regression", "softclass", "quantile"]
    test_size : float, default = 0.1
        The proportion of data to use for the test set.
        The remaining (1 - test_size) of data will be used for the training set.
    random_state : int, default = 0
        Random seed to use during the split.
    min_cls_count_train : int, default = 1
        The minimum number of instances of each class that must occur in the training set (for classification).
        If not satisfied by the original split, instances of unsatisfied classes are
        taken from test and put into train until satisfied.
        Raises an exception if impossible to satisfy.

    Returns
    -------
    train_data, test_data : (DataFrame, DataFrame)
        The train_data and test_data after performing the split. Includes the label column.
    """
    X, y = extract_label(data=data, label=label)
    train_data, test_data, y_train, y_test = generate_train_test_split(
        X=X,
        y=y,
        problem_type=problem_type,
        test_size=test_size,
        random_state=random_state,
        min_cls_count_train=min_cls_count_train)
    train_data[label] = y_train
    test_data[label] = y_test
    return train_data, test_data
