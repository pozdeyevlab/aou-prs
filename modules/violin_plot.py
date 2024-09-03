"""
Module to automate performance evaluation
"""

from pathlib import Path
from typing import List, Optional

import defopt
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


# pylint: disable = C0301
# pylint: disable = R0903 # Too few public methods
# pylint: disable = R1728


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


if __name__ == "__main__":
    defopt.run(prs_violin_plots)

