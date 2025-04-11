# /bin/bash

for dataset in "bc5chem" #"bc5disease" "bc2gm" "chemprotchem" "chemprotgene"
do
    for corruption_type in "random-id-labels" "swapped-id-labels" "random-ood-labels" "random-ood-labels-from-text" "corrupted-ood-text" "corrupted-and-shuffled-ood-text" "corrupted-ood-text-and-labels" "corrupted-and-shuffled-ood-text-and-labels"
    do
        python incontext_ned.py \
            --config-path configs/analysis_random_corrupted_demos/ \
            data="$dataset" \
            corruption_type="$corruption_type" 
    done
done