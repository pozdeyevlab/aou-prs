echo 'running'
# Read in named command line args
while getopts ":e:v:s:o:i:d:m:" opt; do
  case $opt in
    e) escalator="$OPTARG"
    ;;
    s) pgen_suffix="$OPTARG"
    ;;
    o) output_dir="$OPTARG"
    ;;
    i) pgen_dir="$OPTARG"
    ;;
    d) weight_dir="$OPTARG"
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
#Bash masterPRS_v4.sh [reformatting script designed (1, 2, or 3)] \
#[input directory (where weight file is)] \
#[weight input filename] \
#[output directory] \
#[trait name (trait_PGSxxx)] \
#[pfile directory] \
#[pfile prefix name - ex: chr22_freeze3_dosages_PAIR.pgen = freeze3_dosages_PAIR]
#[whether to remove ambiguous variants] \
#[frequency file under the input directory to impute missing genotypes, can be 'NA' if none]

while IFS=" " read -r pgs_id  weight_file file_version;
do
	echo ${escalator}
	echo ${weight_dir}
	echo ${output_dir}
	bash ${escalator} 3 ${weight_dir} ${weight_file} ${output_dir} ${pgs_id} ${pgen_dir} ${pgen_suffix} F NA
done < ${map_file}
