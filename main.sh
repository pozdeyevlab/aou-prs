echo 'running'
# Read in named command line args
while getopts ":d:m:w:p:i:" opt; do
  case $opt in
    d) weight_dir="$OPTARG"
    ;;
    m) map_file="$OPTARG"
    ;;
    w) weight_file="$OPTARG"
    ;;
    p) pgen_dir=$"OPTARG"
    ;;
    i) meta_data="$OPTARG"
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

echo "Starting"

# Format Weight Inputs
echo "Formatting Weights"
bash bash_scripts/make_map.sh -i $weight_file -o $map_file -w $weight_dir

# Run Escalator
echo "Starting ESCALATOR"
bash bash_scripts/escalator.sh \
-e /ESCALATOR/eureka_cloud_version/scripts/masterPRS_v4.sh \
-s filtered_v7 \
-o escalator_output \
-i $pgen_dir \
-d $weight_dir \
-m $map_file

# Run Regression Analysis
echo "Starting Regression"
bash bash_scripts/regression.sh \
-s escalator_output \
-d $meta_data \
-m $map_file

echo "Done"