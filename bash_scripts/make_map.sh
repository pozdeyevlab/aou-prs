# Read in named command line args
while getopts ":i:o:w:" opt; do
  case $opt in
    i) input_file="$OPTARG"
    ;;
    o) output_file="$OPTARG"
    ;;
    w) weight_dir="$OPTARG"
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

mkdir -p $weight_dir
rm -f $output_file

while IFS=" " read -r pgs phenotype regression;
do 
  # Download PGS file
  wget https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/${pgs}/ScoringFiles/Harmonized/${pgs}_hmPOS_GRCh38.txt.gz 

  # Move file to weigth directory
  mv ${pgs}_hmPOS_GRCh38.txt.gz $weight_dir

  # Make ID
  ID=${pgs}_${phenotype}_${regression}

  # Get version
  version=$(gunzip -c $weight_dir/${pgs}_hmPOS_GRCh38.txt.gz | grep 'version' | sed 's/#format_version=//g' | sed 's/.0//g')

  # Get header amt
  header_n=$(gunzip -c $weight_dir/${pgs}_hmPOS_GRCh38.txt.gz | grep -c '^#')

  # Reformat weight file to account for escalator
  python modules/format_weights.py --input-file $weight_dir/${pgs}_hmPOS_GRCh38.txt.gz --header-n $header_n --output-file $weight_dir/${pgs}_formatted_hmPOS_GRCh38.txt
  gzip $weight_dir/${pgs}_formatted_hmPOS_GRCh38.txt

  # Write output files
  echo ${ID} ${pgs}_formatted_hmPOS_GRCh38.txt.gz $version $phenotype $regression >> $output_file

done < $input_file
