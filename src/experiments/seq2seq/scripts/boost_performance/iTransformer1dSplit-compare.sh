folderList=("ECL" "ETT" "Exchange" "PEMS" "SolarEnergy" "Traffic" "Weather")
fileNamesList=("Transformer.sh" "iTransformer.sh" "iTransformer1dSplit.sh")
# iTransformer1dSplit
for folder in "${folderList[@]}"; do
    for fileName in "${fileNamesList[@]}"; do
        echo "Running ${fileName} on ${folder}"
        bash ./scripts/boost_performance/"${folder}"/"${fileName}"
    done
done