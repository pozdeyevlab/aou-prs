"""
Module to automate performance evaluation
"""

import defopt
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# pylint: disable = C0301
# pylint: disable = R0903 # Too few public methods
# pylint: disable = R1728


def linear_regression(*, all_pl: pl.DataFrame, formula: str, disease: str) -> None:
    """
    :param all_pl: Data
    :param formula: Formula
    :param disease: String
    """
    # Subset data to only that required by the formula
    formula_list = formula.replace(" ", "").split("~")[1].split("+")
    formula_list.append(disease)
    all_pl = all_pl.select(formula_list).drop_nulls()

    # Convert columns of type str to dummy values
    str_columns = [col for col in all_pl.columns if all_pl[col].dtype == pl.Utf8]
    
    if len(str_columns) != 0:
        dummy_cols = all_pl.select(str_columns).to_dummies()
        all_pl = all_pl.drop(str_columns)
        all_pl = pl.concat((all_pl, dummy_cols), how="horizontal")
    if all_pl.shape[0] >= 20:
        n = all_pl.shape[0]
        # Collect all columns that are required covariates
        X = all_pl.drop(disease).to_numpy()
        y = all_pl.select(disease).to_numpy()

        # Split data 80:20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Only continue if there are at least 20 people in the test set
        if y.shape[0] > 20:
            linreg = LinearRegression()
            linreg.fit(X, y)
            y_pred = linreg.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Summarize PRS data
            if "SCORE" in formula:
                mean_prs = all_pl["SCORE"].mean()
                var_prs = all_pl["SCORE"].var()
                median_prs = all_pl["SCORE"].median()
                q1 = np.quantile(all_pl["SCORE"].to_numpy(), 0.25)
                q2 = np.quantile(all_pl["SCORE"].to_numpy(), 0.5)
                q3 = np.quantile(all_pl["SCORE"].to_numpy(), 0.75)
            else:
                mean_prs = None
                var_prs = None
                median_prs = None
                q1 = None
                q2 = None
                q3 = None

            return [
                mse,
                r2,
                X_train.shape[0],
                X_test.shape[0],
                mean_prs,
                median_prs,
                var_prs,
                q1,
                q2,
                q3,
                n
            ]
        else:
            return None


if __name__ == "__main__":
    defopt.run(linear_regression)

