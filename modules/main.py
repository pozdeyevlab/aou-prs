"""
Module to automate performance evaluation
"""
import itertools
import sys
import os
from pathlib import Path
from typing import List
import defopt
import polars as pl
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import linear_regression, logistic_regression, violin_plot
# pylint: disable = C0301
# pylint: disable = R0903 # Too few public methods
# pylint: disable = R1728


def auroc(
    *,
    scores_file: Path,
    pgs: str,
    disease_col: str,
    prs_plots: bool,
    demographic_data: Path,
    demo_map: Path,
    output_dir: str,
) -> None:
    """
    :param scores_file: Path to a file (local disc) that contains demographic information as well as prs output
    :param demo_map: TSV mapping the disease name to the phenotype column
    :param binary: T/F Flag for running either logistic or linear regression
    :param pgs: Polygenic score catalog number
    :param disease_col: Name of phenotype
    :param prs_plots: Boolean to plot PRS density or not
    :param demographic_data: TSV file with demographic data accompanied by phenotype hardcalls (binary must be in form of 0 - 1)
    :param output_dir: A path to the output directory you want
    """
    # Get disease_col
    disease_map = pl.read_csv(demo_map, separator='\t', infer_schema_length=10000)
    disease = disease_map.filter(pl.col('from_name')==disease_col)['in_meta_data'].to_list()[0]
    disease_data = disease_map.filter(pl.col('from_name')==disease_col)['meta_df'].to_list()[0]
    binary = disease_map.filter(pl.col('from_name')==disease_col)['regression'].to_list()[0]
    
    print(disease_col)
    print(disease)
    print(disease_data)
    print(binary)
    
    # Read in data
    all_pl = pl.read_csv(scores_file, separator="\t", infer_schema_length=10000, null_values = ['None', 'NA'], schema_overrides={'FID':int, 'IID':int, 'ALLELE_CT':float, 'NAMED_ALLELE_DOSAGE_SUM': float, 'SCORE':float}).select(['IID', 'SCORE'])
    demographics = pl.read_csv(demographic_data, separator='\t', null_values = ['None', 'NA'], infer_schema_length=90000)
    if disease_data != 'fixed_data_april_new_age.tsv':
        # Filter out current age and agesq
        demographics = demographics.drop('age', 'agesq')
        if disease in demographics.columns:
            demographics = demographics.drop(disease)
            print(demographics)
            print(all_pl)

        # Per-disease data 
        disease_df = pl.read_csv(disease_data, separator='\t', null_values = ['NA'], infer_schema_length=10000).select('person_id', 'age_cleaned', 'agesq', disease).rename({'person_id':'IID', 'age_cleaned':'age'})
        demographics = demographics.join(disease_df, on = 'IID', how = 'inner')

    all_pl = all_pl.join(demographics, on = 'IID', how = 'inner')
    print(all_pl.select(disease))
    all_pl = all_pl.filter(pl.col('ethnicity') == 'Not Hispanic or Latino')
    # Define method
    if binary.lower() == 'logistic':
        method = "logistic"
        header = "PGS\tSTRATA_ONE\tGROUP_ONE\tSTRATA_TWO\tGROUP_TWO\tCOVARIATES\tAUC\tTRUE_POSITIVE\tFALSE_POSITIVE\tTRUE_NEGATIVE\tFALSE_NEGATIVE\tDISEASE_PREV\tMEAN_PRS\tMEDIAN_PRS\tVAR_PRS\tPRS_Q1\tPRS_Q2\tPRS_Q3\tCASES\tCONTROLS\n"
        strata_col = 'AUC'
    else:
        method = "linear"
        header = "PGS\tSTRATA_ONE\tGROUP_ONE\tSTRATA_TWO\tGROUP_TWO\tCOVARIATES\tRSQ\tMSE\tTRAINING_SIZE\tTESTING_SIZE\tMEAN_PRS\tMEDIAN_PRS\tVAR_PRS\tPRS_Q1\tPRS_Q2\tPRS_Q3\tN\n"
        strata_col='RSQ'

    # Check output dir exists
    directory = output_dir

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define desired stratifications and make dictionary of strata and subsequent subgroups
    if 'male' in disease:
        stratifications = [
        "race",
        "ethnicity",
        "ancestry_pred",
        "income_quartiles",
        "education",
        "age_quartiles",
        "APOE",
            "Survey_Hysterectomy_History_PMI_Skip",
        "Survey_Hysterectomy_History_Hysterectomy_History_No",
        "Survey_Hysterectomy_History_Hysterectomy_History_Yes",
        "Survey_Hysterectomy_History_PMI_Prefer_Not_To_Answer",
        "Survey_Hysterectomy_History_Hysterectomy_History_Not_Sure",
        "Survey_Menstrual_Stopped_PMI_Skip",
        "Survey_Menstrual_Stopped_PMI_Prefer_Not_To_Answer",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Yes_None",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Yes_But_Hormone",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Periods_Havent_Stopped",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Not_Sure_Menstrual_Stopped",
        "Survey_Ovary_Removal_History_PMI_Skip",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_No",
        "Survey_Ovary_Removal_History_PMI_Prefer_Not_To_Answer",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Not_Sure",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Both",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Unsure",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Sectioned",
        "Survey_Menstrual_Stopped_Reason_PMI_Skip",
        "Survey_Menstrual_Stopped_Reason_PMI_Other",
        "Survey_Menstrual_Stopped_Reason_PMI_Not_Sure",
        "Survey_Menstrual_Stopped_Reason_PMI_Prefer_Not_To_Answer",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Surgery",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Natural_Menopause",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Medication_Therapy",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Endometrial_Ablation",
        "EHR_SNOMED_237788002_Premature_ovarian_failure",
        "EHR_SNOMED_65846009_Primary_ovarian_failure",
        "EHR_SNOMED_373717006_Premature_menopause",
        "EHR_SNOMED_289903006_Menopause_present",
        "EHR_SNOMED_237138004_Menopause_ovarian_failure",
        "EHR_SNOMED_70670009_Abnormal_vasomotor_function"]
    if 'male' not in disease:
        stratifications = [
        "race",
        "ethnicity",
        "sex_at_birth",
        "ancestry_pred",
        "income_quartiles",
        "education",
        "gender",
        "age_quartiles",
        "APOE",
        "Survey_Hysterectomy_History_PMI_Skip",
        "Survey_Hysterectomy_History_Hysterectomy_History_No",
        "Survey_Hysterectomy_History_Hysterectomy_History_Yes",
        "Survey_Hysterectomy_History_PMI_Prefer_Not_To_Answer",
        "Survey_Hysterectomy_History_Hysterectomy_History_Not_Sure",
        "Survey_Menstrual_Stopped_PMI_Skip",
        "Survey_Menstrual_Stopped_PMI_Prefer_Not_To_Answer",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Yes_None",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Yes_But_Hormone",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Periods_Havent_Stopped",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Not_Sure_Menstrual_Stopped",
        "Survey_Ovary_Removal_History_PMI_Skip",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_No",
        "Survey_Ovary_Removal_History_PMI_Prefer_Not_To_Answer",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Not_Sure",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Both",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Unsure",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Sectioned",
        "Survey_Menstrual_Stopped_Reason_PMI_Skip",
        "Survey_Menstrual_Stopped_Reason_PMI_Other",
        "Survey_Menstrual_Stopped_Reason_PMI_Not_Sure",
        "Survey_Menstrual_Stopped_Reason_PMI_Prefer_Not_To_Answer",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Surgery",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Natural_Menopause",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Medication_Therapy",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Endometrial_Ablation",
        "EHR_SNOMED_237788002_Premature_ovarian_failure",
        "EHR_SNOMED_65846009_Primary_ovarian_failure",
        "EHR_SNOMED_373717006_Premature_menopause",
        "EHR_SNOMED_289903006_Menopause_present",
        "EHR_SNOMED_237138004_Menopause_ovarian_failure",
        "EHR_SNOMED_70670009_Abnormal_vasomotor_function"]
    if disease == 'college_education':
        stratifications = [
        "race",
        "ethnicity",
        "sex_at_birth",
        "ancestry_pred",
        "income_quartiles",
        "gender",
        "age_quartiles",
        "APOE",
        "Survey_Hysterectomy_History_PMI_Skip",
        "Survey_Hysterectomy_History_Hysterectomy_History_No",
        "Survey_Hysterectomy_History_Hysterectomy_History_Yes",
        "Survey_Hysterectomy_History_PMI_Prefer_Not_To_Answer",
        "Survey_Hysterectomy_History_Hysterectomy_History_Not_Sure",
        "Survey_Menstrual_Stopped_PMI_Skip",
        "Survey_Menstrual_Stopped_PMI_Prefer_Not_To_Answer",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Yes_None",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Yes_But_Hormone",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Periods_Havent_Stopped",
        "Survey_Menstrual_Stopped_Menstrual_Stopped_Not_Sure_Menstrual_Stopped",
        "Survey_Ovary_Removal_History_PMI_Skip",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_No",
        "Survey_Ovary_Removal_History_PMI_Prefer_Not_To_Answer",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Not_Sure",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Both",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Unsure",
        "Survey_Ovary_Removal_History_Ovary_Removal_History_Yes_Sectioned",
        "Survey_Menstrual_Stopped_Reason_PMI_Skip",
        "Survey_Menstrual_Stopped_Reason_PMI_Other",
        "Survey_Menstrual_Stopped_Reason_PMI_Not_Sure",
        "Survey_Menstrual_Stopped_Reason_PMI_Prefer_Not_To_Answer",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Surgery",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Natural_Menopause",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Medication_Therapy",
        "Survey_Menstrual_Stopped_Reason_Menstrual_Stopped_Reason_Endometrial_Ablation",
        "EHR_SNOMED_237788002_Premature_ovarian_failure",
        "EHR_SNOMED_65846009_Primary_ovarian_failure",
        "EHR_SNOMED_373717006_Premature_menopause",
        "EHR_SNOMED_289903006_Menopause_present",
        "EHR_SNOMED_237138004_Menopause_ovarian_failure",
        "EHR_SNOMED_70670009_Abnormal_vasomotor_function"]
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
            "other"
        ]

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
    file_path = os.path.join(directory, f"{pgs}_{disease_col}.tsv")
    # Open file to append
    with open(file_path, "w+") as f:
        f.write(header)
        for combo in refined_combinations:
            print(combo)
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
                    results = linear_regression.linear_regression(all_pl=subset_data, formula=formula[0], disease=disease)
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
                        n = results[10]
                        f.write(
                            f"{pgs}\t{strata_one}\t{group_one}\t{strata_two}\t{group_two}\t{formula[1]}\t{r2}\t{mse}\t{train_n}\t{test_n}\t{mean_prs}\t{median_prs}\t{var_prs}\t{q1}\t{q2}\t{q3}\t{n}\n"
                        )
            if method == "logistic":
                for formula in formula_zipped:
                    results = logistic_regression.logistic_regression(
                        all_pl=subset_data,
                        formula=formula[0],
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
                        cases = results[12]
                        controls =results[13]
                        f.write(
                            f"{pgs}\t{strata_one}\t{group_one}\t{strata_two}\t{group_two}\t{formula[1]}\t{auc}\t{tp}\t{fp}\t{tn}\t{fn}\t{disease_prev}\t{mean_prs}\t{median_prs}\t{var_prs}\t{q1}\t{q2}\t{q3}\t{cases}\t{controls}\n"
                        )

    incremental(results=file_path, strata_col=strata_col)


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
    df = pl.read_csv(results, separator="\t", null_values=['None'], infer_schema_length=10000)
    # Group by 'GROUP' and 'STRATA' and aggregate stat values
    for name, data in df.group_by(
        ["GROUP_ONE", "STRATA_ONE", "GROUP_TWO", "STRATA_TWO"]
    ):
        both = data.filter(pl.col("COVARIATES") == "prs_and_covariates")[strata_col]
        cov_only = data.filter(pl.col("COVARIATES") == "covariates_only")[strata_col]
        incremental = both - cov_only
        
        try:
            data = data.with_columns(
                (pl.lit(incremental[0])).alias(f"INCREMENTAL_{strata_col}_PRS")
            )

        except IndexError:
            data = data.with_columns(
                (pl.lit(0)).alias(f"INCREMENTAL_{strata_col}_PRS")
            )


        dfs.append(data)

    new_df = pl.concat(dfs)
    # Write over the old file
    new_df.write_csv(results, separator="\t")
    print(f"Success {results}")


if __name__ == "__main__":
    defopt.run(auroc)

