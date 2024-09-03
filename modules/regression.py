"""
Module to automate performance evaluation
"""
import sys
import itertools
import os
from pathlib import Path
from typing import List, Optional

import defopt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# pylint: disable = C0301
# pylint: disable = R0903 # Too few public methods
# pylint: disable = R1728


def auroc(
    *,
    scores_file: Path,
    binary: str,
    pgs: str,
    disease: str,
    prs_plots: bool,
    demographic_data: Path,
) -> None:
    """
    :param scores_file: Path to a file (local disc) that contains demographic information as well as prs output
    :param binary: T/F Flag for running either logistic or linear regression
    :param pgs: Polygenic score catalog number
    :disease: Name of phenotype
    :param prs_plots: Boolean to plot PRS density or not
    :param demographic_data: TSV file with demographic data accompanied by phenotype hardcalls (binary must be in form of 0 - 1)
    """
    # Read in data
    all_pl = pl.read_csv(scores_file, separator="\t").select(['IID', 'SCORE'])
    demographics = pl.read_csv(demographic_data, separator='\t')
    all_pl = all_pl.join(demographics, on = 'IID', how = 'inner')

    # Define method
    if binary.lower() == 'logistic':
        method = "logistic"
        header = "PGS\tSTRATA_ONE\tGROUP_ONE\tSTRATA_TWO\tGROUP_TWO\tCOVARIATES\tAUC\tTRUE_POSITIVE\tFALSE_POSITIVE\tTRUE_NEGATIVE\tFALSE_NEGATIVE\tDISEASE_PREV\tMEAN_PRS\tMEDIAN_PRS\tVAR_PRS\tPRS_Q1\tPRS_Q2\tPRS_Q3\n"
        strata_col = 'AUC'
    else:
        method = "linear"
        header = "PGS\tSTRATA_ONE\tGROUP_ONE\tSTRATA_TWO\tGROUP_TWO\tCOVARIATES\tRSQ\tMSE\tTRAINING_SIZE\tTESTING_SIZE\tMEAN_PRS\tMEDIAN_PRS\tVAR_PRS\tPRS_Q1\tPRS_Q2\tPRS_Q3\n"
        strata_col='RSQ'

    # Check output dir exists
    directory = f"prs_results/{pgs}_{disease}_{method}_regression_results"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if plotting directories exist otherwise create them
    if prs_plots:
        plot_dir = f"prs_results/{pgs}_{disease}_plots"

        # Create the directory if it doesn't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    else:
        plot_dir = None

    # Define desired stratifications and make dictionary of strata and subsequent subgroups
    stratifications = [
        "race",
        "ethnicity",
        "sex_at_birth",
        "ancestry_pred",
        "income_quartiles",
        "education",
        "gender",
        "age_quartiles",
    ]

    data = {}
    for strata in stratifications:
        # Create groups to loop through
        groups = set(all_pl[strata].drop_nulls())
        # Remove individuals with ambiguous responses
        bad_columns = [
            "PMI: Prefer Not To Answer",
            "PMI: Skip",
            "I prefer not to answer",
            "None",
            "Intersex",
            "No matching concept",
            "None of these",
            "More than one population",
            "None Indicated",
        ]

        # Call for violin plot per strata
        if prs_plots:
            if method == "logistic":
                hue = disease
            else:
                hue = None

            prs_violin_plots(
                prs_df=all_pl,
                plot_directory=plot_dir,
                pgs=pgs,
                disease=disease,
                strata=strata,
                bad_groups=bad_columns,
                hue=hue,
            )
        to_remove = []
        [to_remove.append(g) for g in groups if g in bad_columns]
        for r in to_remove:
            groups.remove(r)
        for group in groups:
            if strata in data:
                data[strata].append(group)
            if strata not in data:
                data[strata] = [group]

    # Generate combined combinations
    combined_combinations = generate_combinations(data)
    refined_combinations = []
    for combo in combined_combinations:
        gender_check = "gender" in combo and "sex_at_birth" in combo
        ancestry_check = (
            ("ancestry_pred" in combo and "ethnicity" in combo)
            or ("ancestry_pred" in combo and "race" in combo)
            or ("race" in combo and "ethnicity" in combo)
        )

        if not gender_check and not ancestry_check:
            refined_combinations.append(combo)

    # Define the file path within the directory
    file_path = os.path.join(directory, f"{pgs}_{method}_regression.tsv")
    # Open file to append
    with open(file_path, "w+") as f:
        f.write(header)
        for combo in refined_combinations:
            if len(combo) > 2:
                strata_one = combo[0]
                strata_two = combo[2]
                group_one = combo[1]
                group_two = combo[3]
                subset_data = all_pl.filter((pl.col(strata_one) == group_one)).filter(
                    (pl.col(strata_two) == group_two)
                )
                sex_gender_check = (
                    ("gender" in strata_one)
                    or ("sex" in strata_one)
                    or ("gender" in strata_two)
                    or ("sex" in strata_two)
                )
            if len(combo) == 2:
                strata_one = combo[0]
                group_one = combo[1]
                strata_two = None
                group_two = None
                subset_data = all_pl.filter((pl.col(strata_one) == group_one))
                sex_gender_check = ("gender" in strata_one) or ("sex" in strata_one)

            if sex_gender_check:
                # Only include sex asa covariate if not stratifying by sex or gender
                formulas = [
                    f"{disease} ~ SCORE",
                    f"{disease} ~ SCORE + age + agesq +{' + '.join([f'pca_{i}' for i in range(1,17)])}",
                    f"{disease} ~ age + agesq +{' + '.join([f'pca_{i}' for i in range(1, 17)])}",
                ]
            else:
                formulas = [
                    f"{disease} ~ SCORE",
                    f"{disease} ~ SCORE + age + agesq +sex_at_birth + {' + '.join([f'pca_{i}' for i in range(1, 17)])}",
                    f"{disease} ~ age + agesq + sex_at_birth + {' + '.join([f'pca_{i}' for i in range(1, 17)])}",
                ]
            classification = ["prs_only", "prs_and_covariates", "covariates_only"]
            formula_zipped = list(zip(formulas, classification))

            if method == "linear":
                for formula in formula_zipped:
                    results = lin_pred(all_pl=subset_data, prs=pgs, formula=formula[0], disease=disease)
                    if results is not None:
                        mse = results[0]
                        r2 = results[1]
                        train_n = results[2]
                        test_n = results[3]
                        mean_prs = results[4]
                        median_prs = results[5]
                        var_prs = results[6]
                        q1 = results[7]
                        q2 = results[8]
                        q3 = results[9]
                        f.write(
                            f"{pgs}\t{strata_one}\t{group_one}\t{strata_two}\t{group_two}\t{formula[1]}\t{r2}\t{mse}\t{train_n}\t{test_n}\t{mean_prs}\t{median_prs}\t{var_prs}\t{q1}\t{q2}\t{q3}\n"
                        )
            if method == "logistic":
                for formula in formula_zipped:
                    results = log_pred(
                        all_pl=subset_data,
                        prs=pgs,
                        formula=formula[0],
                        plot_directory=plot_dir,
                        classification=formula[1],
                        disease=disease
                    )
                    if results is not None:
                        auc = results[0]
                        tp = results[1]
                        fp = results[2]
                        tn = results[3]
                        fn = results[4]
                        disease_prev = results[5]
                        mean_prs = results[6]
                        median_prs = results[7]
                        var_prs = results[8]
                        q1 = results[9]
                        q2 = results[10]
                        q3 = results[11]
                        f.write(
                            f"{pgs}\t{strata_one}\t{group_one}\t{strata_two}\t{group_two}\t{formula[1]}\t{auc}\t{tp}\t{fp}\t{tn}\t{fn}\t{disease_prev}\t{mean_prs}\t{median_prs}\t{var_prs}\t{q1}\t{q2}\t{q3}\n"
                        )

    incremental(results=file_path, strata_col=strata_col)

def lin_pred(*, all_pl: pl.DataFrame, prs: str, formula: str, disease: str) -> None:
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
        print(all_pl.columns)
        sys.exit(1)
    if all_pl.shape[0] >= 100:
        # Collect all columns that are required covariates
        X = all_pl.drop(disease).to_numpy()
        y = all_pl.select(disease).to_numpy()

        # Split data 80:20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Only continue if there are at least 20 people in the test set
        if y_test.shape[0] > 20:
            linreg = LinearRegression()
            linreg.fit(X_train, y_train)
            y_pred = linreg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

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
            ]
        else:
            return None


def log_pred(
    *,
    all_pl: pl.DataFrame,
    formula: str,
    disease: str,
) -> None:
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
    if all_pl.shape[0] > 100:
        # Collect all columns that are required covariates
        X = all_pl.drop(disease).to_numpy()
        y = all_pl.select(disease).to_numpy()

        # Only continue if there are atleast 20 positive people in the test set
        if y.sum() > 20:
            logreg = LogisticRegression(random_state=10)  # class_weight="balanced")
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
            ]
    else:
        return None


def prs_violin_plots(
    *,
    prs_df: pl.DataFrame,
    plot_directory: Path,
    pgs: str,
    strata: str,
    bad_groups=List[str],
    hue: Optional[str],
    disease: str,
) -> None:
    # Make dataframe
    prs_df = (
        prs_df.select(["SCORE", strata, disease])
        .with_columns(
            pl.when(pl.col(strata).str.contains_any(bad_groups))
            .then(None)
            .otherwise(pl.col(strata))
            .alias(strata)
        )
        .drop_nulls()
    )

    # Filter for counts less than 20 individuals
    if hue is not None:
        keep = (
            prs_df.group_by(strata)
            .agg(pl.col(disease).sum())
            .filter(pl.col(disease) > 20)
        )[strata].to_list()

    if hue is None:
        keep = (
            prs_df.select(strata)
            .value_counts()
            .filter(pl.col("count") > 20)[strata]
            .to_list()
        )

    prs_df = (
        prs_df.select(["SCORE", strata, disease])
        .with_columns(
            pl.when(~(pl.col(strata).str.contains_any(keep)))
            .then(None)
            .otherwise(pl.col(strata))
            .alias(strata)
        )
        .drop_nulls()
    )

    # Make Violin Plot
    plt.figure(figsize=(10, 10))
    sns.set(style="darkgrid")
    sns.hls_palette()
    if hue is not None:
        plot = sns.violinplot(
            data=prs_df.to_pandas(),
            y="SCORE",
            x=strata,
            hue=disease,
            palette="hls",
            split=False,
            gap=0.3,
            inner_kws=dict(box_width=40, whis_width=2),
        )
        sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
        plot.set_title(f"{disease}: {pgs} PRS Distribution")

    if hue is None:
        plot = sns.violinplot(
            data=prs_df.to_pandas(), y="SCORE", x=strata, palette="hls"
        ).set_title(f"{disease}: {pgs} PRS Distribution")

    plt.figure(figsize=(10, 60))
    plt.tight_layout()
    fig = plot.get_figure()
    fig.autofmt_xdate()
    fig.savefig(f"{plot_directory}/{pgs}_{strata}_violin_plot.png")


def generate_combinations(data: dict) -> List:
    # Generate individual key-value pair combinations
    individual_combinations = []
    for key, values in data.items():
        for value in values:
            individual_combinations.append((key, value))

    # Generate all pairs of keys for intersections
    key_pairs = list(itertools.combinations(data.keys(), 2))

    # Generate intersections for each pair of keys
    formatted_intersections = []
    for key1, key2 in key_pairs:
        values_product = list(itertools.product(data[key1], data[key2]))
        for combination in values_product:
            formatted_intersections.append((key1, combination[0], key2, combination[1]))

    # Combine both individual combinations and formatted intersections into a single list
    combined_combinations = individual_combinations + formatted_intersections

    return combined_combinations


def incremental(results: Path, strata_col: str) -> None:
    dfs = []
    # Add incremental RSQ or AUC calculations
    df = pl.read_csv(results, separator="\t")
    # Group by 'GROUP' and 'STRATA' and aggregate stat values
    for name, data in df.group_by(
        ["GROUP_ONE", "STRATA_ONE", "GROUP_TWO", "STRATA_TWO"]
    ):
        both = data.filter(pl.col("COVARIATES") == "prs_and_covariates")[strata_col]
        cov_only = data.filter(pl.col("COVARIATES") == "covariates_only")[strata_col]
        incremental = both - cov_only
        data = data.with_columns(
            (incremental).alias(f"INCREMENTAL_{strata_col}_PRS")
        )
        dfs.append(data)

    new_df = pl.concat(dfs)
    # Write over the old file
    new_df.write_csv(results, separator="\t")
    print(f"Success {results}")


if __name__ == "__main__":
    defopt.run(auroc)

