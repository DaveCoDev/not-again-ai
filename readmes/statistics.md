# Statistics
The statistics module provides functions to assist with various statistical computations leveraging popular libraries like `numpy`, `scipy`, and `sklearn`.

## Dependence
### `pearson_correlation`
Computes the absolute value of the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) using [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html). It automatically handles both numerical and categorical variables. If the second variable contains all the same values, a coefficient of `1` is returned.

### `pred_power_score_classification`
This function computes the Predictive Power Score (PPS) inspired by [ppscore](https://github.com/8080labs/ppscore). It is an asymmetric score that identifying both linear and non-linear relationships between two variables. The score is computed for a classification task and requires the second variable to be categorical. If the second variable has identical values throughout, the function returns a score of `1`.
