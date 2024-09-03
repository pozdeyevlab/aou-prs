echo 'running'
# Read in named command line args
while getopts ":s:o:i:d:m:" opt; do
  case $opt in
    s) score_dir="$OPTARG"
    ;;
    d) demographic_file="$OPTARG"
    ;;
    m) map_file="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

# Arguments
#usage: regression.py [-h] -s SCORES_FILE -b BINARY --pgs PGS --disease DISEASE --prs-plots | #--no-prs-plots
#                     --demographic-data DEMOGRAPHIC_DATA
#
#options:
#  -h, --help            show this help message and exit
#  -s SCORES_FILE, --scores-file SCORES_FILE
#                        Path to a file (local disc) that contains demographic information as well #as prs output
#  -b BINARY, --binary BINARY
#                        T/F Flag for running either logistic or linear regression
#  --pgs PGS             Polygenic score catalog number
#  --disease DISEASE
#  --prs-plots, --no-prs-plots
#                        Boolean to plot PRS density or not
#  --demographic-data DEMOGRAPHIC_DATA
#                        TSV file with demographic data accompanied by phenotype hardcalls (binary #must be in form of 0 - 1)

while IFS=" " read -r pgs_id  weight_file file_version disease binary;
do      
    # attach scores directory to file name
    score_file=${score_dir}/${pgs_id}_prs.sscore
    pgs=$(echo ${pgs_id} |cut -d'_' -f1)
    echo ${pgs}
    python modules/main.py --binary ${binary} --scores-file ${score_file} --pgs ${pgs} --disease ${disease} --no-prs-plots --demographic-data ${demographic_file}
done < ${map_file}
