"""
Helper for formatting input files
"""

import sys
from pathlib import Path
from typing import List

import defopt
import polars as pl

# pylint: disable=R0914, R0913, R0903, C0301


def format(*, input_file: str, output_file: Path, header_n: int) -> pl.DataFrame:
    """
    :param input_file: Input weight file
    :param output_file: Output file
    :param header_n: Number of lines that start with '#'
    """
    try:
        data = pl.read_csv(
            input_file, skip_rows=(header_n), separator="\t", infer_schema_length=0
            ).drop('hm_chr').rename({'chr_name':'hm_chr'}).select(["hm_chr", "hm_pos", "effect_allele", "other_allele", "effect_weight"])
        header = pl.read_csv(
            input_file,
            n_rows=(header_n),
            separator="\t",
            infer_schema_length=0,
            truncate_ragged_lines=True,
            has_header=False,
        )
    except:
        data = pl.read_csv(
            input_file, skip_rows=(header_n), separator="\t", infer_schema_length=0
            ).drop('hm_chr').rename({'chr_name':'hm_chr'}).select(["hm_chr", "hm_pos", "effect_allele", "hm_inferOtherAllele", "effect_weight"]).rename({'hm_inferOtherAllele':'other_allele'})
        header = pl.read_csv(
            input_file,
            n_rows=(header_n),
            separator="\t",
            infer_schema_length=0,
            truncate_ragged_lines=True,
            has_header=False,
        )

    print(data)
    print(header)

    # Open the file in write mode
    with open(output_file, "w") as f:
        # Write rows from the first DataFrame
        for row in header.iter_rows():
            f.write("\t".join(map(str, row)) + "\n")

        # Write the header for the second DataFrame
        f.write("\t".join(data.columns) + "\n")

        # Write rows from the second DataFrame
        for row in data.iter_rows():
            f.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    defopt.run(format)
