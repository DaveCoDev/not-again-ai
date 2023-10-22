import numpy as np
import numpy.typing as npt
import scipy
import sklearn.metrics as skmetrics
import sklearn.model_selection as skmodel_selection
import sklearn.preprocessing as skpreprocessing
import sklearn.tree as sktree


def _process_variable(
    x: npt.NDArray[np.int_] | (npt.NDArray[np.float_] | npt.NDArray[np.str_]),
) -> npt.NDArray[np.int_] | (npt.NDArray[np.float_] | npt.NDArray[np.str_]):
    """Process variable by encoding it as a numeric array."""
    le = skpreprocessing.LabelEncoder()
    x = le.fit_transform(x)
    return x


def pearson_correlation(
    x: list[int]
    | (list[float] | (list[str] | (npt.NDArray[np.int_] | (npt.NDArray[np.float_] | npt.NDArray[np.str_])))),
    y: list[int]
    | (list[float] | (list[str] | (npt.NDArray[np.int_] | (npt.NDArray[np.float_] | npt.NDArray[np.str_])))),
    is_x_categorical: bool = False,
    is_y_categorical: bool = False,
    print_diagnostics: bool = False,
) -> float:
    """Absolute value of the Pearson correlation coefficient.
    Returns 1 in the case y contains all of the same values.

    Implemented using scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Args:
        x (listlike): first variable
        y (listlike): second variable
        is_x_categorical (bool): whether x is categorical
        is_y_categorical (bool): whether y is categorical
        print_diagnostics (bool): whether to print diagnostics to stdout
    """
    x_array = np.array(x)
    y_array = np.array(y)

    if is_x_categorical:
        x_array = _process_variable(x_array)

    if is_y_categorical:
        y_array = _process_variable(y_array)

    # check if y contains all of the same values, and if so return 1
    if len(np.unique(y_array)) == 1:
        if print_diagnostics:
            print("y contains all of the same values, returning 1")
        return 1.0

    pearsonr = scipy.stats.pearsonr(x_array, y_array)
    metric: float = pearsonr.statistic
    metric = np.abs(metric)
    return metric


def pred_power_score_classification(
    x: list[int]
    | (list[float] | (list[str] | (npt.NDArray[np.int_] | (npt.NDArray[np.float_] | npt.NDArray[np.str_])))),
    y: list[int] | (list[str] | npt.NDArray[np.int_]),
    cv_splits: int = 5,
    print_diagnostics: bool = False,
) -> float:
    """Compute Predictive Power Score, an asymmetric score that can detect
    linear or non-linear relationships between two variables.
    For this implementation, the score is computed for a classification task and y must be categorical.

    Returns 1 in the case y contains all of the same values.

    Args:
        x (listlike of int, float, or string): first variable
        y (listlike of int or string): second variable
        cv_splits (int): number of cross-validation splits
        print_diagnostics (bool): whether to print diagnostics to stdout
    """
    x_array = np.array(x)
    y_array = np.array(y)

    le = skpreprocessing.LabelEncoder()
    # check if x contains any strings
    if any(isinstance(elem, str) for elem in x_array):
        x_array = le.fit_transform(x_array)

    x_array = x_array.reshape(-1, 1)
    y_array = le.fit_transform(y_array)

    # check if y contains all of the same values, and if so return 1
    if len(np.unique(y_array)) == 1:
        if print_diagnostics:
            print("y contains all of the same values, returning 1")
        return 1.0

    # Use KFold cross-validation to compute weighted (macro) F1 score
    model = sktree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, random_state=0)
    cv_method = skmodel_selection.KFold(n_splits=cv_splits, shuffle=True, random_state=0)
    f1_scores = skmodel_selection.cross_val_score(
        model, x_array, y_array, cv=cv_method, scoring="f1_weighted", error_score="raise"
    )
    f1 = np.mean(f1_scores)

    # find majority class in y
    majority_class = np.argmax(np.bincount(y_array))
    preds = np.ones_like(y_array) * majority_class
    f1_null: float = skmetrics.f1_score(y_array, preds, average="weighted")

    # generate random predictions
    preds = np.random.choice(np.unique(y_array), size=len(y_array))
    f1_random: float = skmetrics.f1_score(y_array, preds, average="weighted")

    f1_naive = np.max([f1_null, f1_random])
    pps: float = (f1 - f1_naive) / (1 - f1_naive)

    # ensure pps is not negative
    pps = np.max([0, pps])
    return pps
