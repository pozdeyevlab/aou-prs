"""
Module to automate performance evaluation
"""

import defopt
import numpy as np
import polars as pl
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# pylint: disable = C0301
# pylint: disable = R0903 # Too few public methods
# pylint: disable = R1728


def logistic_regression(
    *,
    all_pl: pl.DataFrame,
    formula: str,
    disease: str,
) -> None:
    """
    :param all_pl: Data
    :param formula: Formula
    :param disease: Disease
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
    if all_pl.shape[0] > 20:
        # Collect all columns that are required covariates
        X = all_pl.drop(disease).to_numpy()
        y = all_pl.select(disease).to_numpy()

        # Only continue if there are atleast 20 positive people in the test set
        if y.sum() > 20:
            cases = all_pl.filter(pl.col(disease) == 1).shape[0]
            controls = all_pl.filter(pl.col(disease) == 0).shape[0]
            logreg = LogisticRegression(random_state=10, max_iter=1000)  # class_weight="balanced")
            logreg.fit(X, y.ravel())

            # Predict probabilities on test set
            y_pred = logreg.predict(X)
   
            # Get confusion matrix
            conf_matrix = confusion_matrix(y, y_pred)
            tn = conf_matrix[0, 0] if conf_matrix[0, 0] > 20 else None
            fp = conf_matrix[0, 1] if conf_matrix[0, 1] > 20 else None
            fn = conf_matrix[1, 0] if conf_matrix[1, 0] > 20 else None
            tp = conf_matrix[1, 1] if conf_matrix[1, 1] > 20 else None
            disease_prev = all_pl[disease].sum() / all_pl.shape[0]
            # Calculate AUC
            auc = metrics.roc_auc_score(y, y_pred)
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
                auc,
                tp,
                fp,
                tn,
                fn,
                disease_prev,
                mean_prs,
                median_prs,
                var_prs,
                q1,
                q2,
                q3,
                cases,
                controls
            ]
    else:
        return None


if __name__ == "__main__":
    defopt.run(logistic_regression)
