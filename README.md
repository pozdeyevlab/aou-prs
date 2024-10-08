# aou-prs
This repository contains the scripts necessary to generate polygenic risk scores (PRS’s) on an [All of Us](https://www.researchallofus.org/) (AoU) persistent disk, using [ESCALATOR](https://github.com/menglin44/ESCALATOR), and then evaluate the performance of the PRS via AUC or R2 measurement. 

**Due to install limitations on AoU, this relies on bash scripts as opposed to a proper workflow language**

# How to use
## AoU Disclaimer
Do not attempt to run the scripts in this repository without having gained 'Controlled Tier Access' on AoU. Additionally, downloading any data pertaining to fewer than 20 individuals is strictly prohibited. By default, prs performance will only be evaluated on cohorts that have at least 20 positive cases for binary phenotypes and 20 total samples for continuous phenotypes. The Pozdeyev Research Group recommends further filtering to ~100 individuals before downloading. This ensures that the data policy will not be downloaded and helps remove potential false positives/negatives that may be artifacts of low statistical power. 

## Setting Up ESCALATOR
Follow the steps laid out below in order to run [ESCALATOR](https://github.com/menglin44/ESCALATOR)
```bash
# Clone this repo
git clone https://github.com/pozdeyevlab/aou-prs.git
cd aou-prs

# Clone ESCALATOR
git clone https://github.com/MatthewFisher126/ESCALATOR.git
tar -xvzf ESCALATOR/eureka_cloud_version/bin/prs_pipeline_bin.tar.gz -C ESCALATOR/eureka_cloud_version/bin/

# Ammend main ESCALATOR script (ESCALATOR/eureka_cloud_version/scripts/masterPRS_v4.sh)
# Replace line 13 of with script_path="ESCALATOR/eureka_cloud_version/scripts"
# Replace line 14 with bin_path="ESCALATOR/eureka_cloud_version/bin/prs_pipeline_bin"
# Hash out lines 83 & 84

# Change python path
sed -i.bak 's/\/bin\/python3/python/g' ESCALATOR/eureka_cloud_version/scripts/masterPRS_v4.sh

# Give plink execute permissions
chmod u+x ESCALATOR/eureka_cloud_version/bin/prs_pipeline_bin/plink2_mar
```
## Required Inputs
1)	**PGS Inputs**:Space separated file with the following columns [pgs (ID), phenotype (disease name), regression (logistic or linear)]
2)	**Meta Data**:Demographics file, a tab separated file with AoU IIDs, sex at birth, education quartiles, income quartiles, age, disease outcomes correlating to disease names in file listed above. 
3)	**Escalator/Regression Map file**:Map file generated via `make_map.sh`
4)	PGENS chr1-22

### Environment & Dependency Set Up
```bash
pip install polars
pip install defopt
```
### Step 1
The first step is to download specified PGS weight files from the PGS Catalog Database, and then ensure they are formatted properly for [escalator](https://github.com/menglin44/ESCALATOR). 
```bash
bash make_map.sh -i <**PGS Inputs**> -o  <**name of map file**> -w <directory to store weight files in> 
```
### Step 2
After downloading and preparing your weight file(s) you are now ready to calculate PRS’s. This is done through `escalator.sh` and requires the following inputs:
-e: Path to masterPRS_v4.sh (escalator script)
-s Suffix of pgens
-o output directory 
-i path to pgen directory
-d path to directory with weight files
-m path to map file, space separated with the following columns ID, weight file, version

```bash
bash escalator.sh \
-e <path to ~/ESCALATOR/eureka_cloud_version/scripts/masterPRS_v4.sh> \
-s <filtered_v7> \
-o <~/pgs_scores> \
-i <~/PGENS> \
-d <~/weights> \
-m <map file>
```
### Step 3
The final step, after escalator has successfully completed, is to run linear/logistic regression on the PRS results. This is orchestrated through `regression.sh`. 
-s path to escalator results directory
-d path to demographics file
-m map file [ID, weight file name, version number, disease]
```bash
bash bash_scripts/regression.sh \
-s <score directory> \
-d <demographic file> \
-m <map file> \
```

# Example 
Example of all three steps (assumes demographic file is named `meta_data.tsv` and pgens have the suffix `filtered_v7`)
```bash
# Format Weight Inputs
bash make_map.sh -i example_weights_input.tsv -o  test_map.tsv -w ./weight_files/

# Run Escalator
bash bash_scripts/escalator.sh \
-e /ESCALATOR-main/eureka_cloud_version/scripts/masterPRS_v4.sh \
-s filtered_v7 \
-o escalator_output \
-i /PGENS/ \
-d /weight_files \
-m test_map.tsv

# Run Regression Analysis
bash bash_scripts/regression.sh \
-s escalator_output \
-d meta_data.tsv \
-m test_map.tsv
```

## Option to use main.sh
```bash
bash main.sh \
-d weights \ # path to weight directory
-m map_file.tsv \ # path to map file
-w weights.txt \ # path to weight inputs
-p pgens \ # path to pgen directory
-i meta_data.tsv \ # path to meta data
```
