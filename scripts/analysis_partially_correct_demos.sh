# /bin/bash

for dataset in "bc5chem" "bc5disease" "bc2gm" "chemprotchem" "chemprotgene"
do
    for perturbation_type in "deletion" "substitution" "deletion-substitution" "addition-substitution"
    do
        for perturbation_factor in "01" "02" "03" "04" "05" "06" "07" "08" "09"
        do
            python incontext_ned.py --config-path configs/analysis_partially_correct_demos/ data="$dataset" perturbation_type="$perturbation_type" perturbation_factor="$perturbation_factor"
        done
    done
done