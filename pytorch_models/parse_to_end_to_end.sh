#!/bin/bash

MODEL=$1
DATASET=$2
AUG=$3
PATH_END_TO_END=end_to_end/data/outputs_json/
ALL_PAGES=(1 2 3 4 5 6 7 8 16 32 64)


for PAGES in ${ALL_PAGES[*]}
do
    for SPLIT in train val test
    do
        mkdir -p end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}
        find ${PATH_END_TO_END}/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT} -name 'pred*json' | sort > "end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}/pred-data.dat"
        find ${PATH_END_TO_END}/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT} -name 'gt*json'   | sort > "end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}/gt-data.dat"
        
        python parse_data_end_to_end.py \
            end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}/pred-data.dat \
            end_to_end/Folds/${DATASET}/fold0/${SPLIT}.txt \
            end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}/pred.dat

        python parse_data_end_to_end.py \
            end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}/gt-data.dat \
            end_to_end/Folds/${DATASET}/fold0/${SPLIT}.txt \
            end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}/gt.dat

        rm end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}/pred-data.dat \
           end_to_end/data/${MODEL}/${AUG}/${DATASET}/fold0/${PAGES}_pages/${SPLIT}/gt-data.dat



    done
done

