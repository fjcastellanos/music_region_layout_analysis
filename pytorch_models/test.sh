#!/bin/bash

MODELS=$1 #(Faster-RCNN MobileNet RetinaNet SSD-VGG16) #$1 #SSD-VGG16 #SSD-VGG16 #RetinaNet #MobileNet #RetinaNet MobileNet SSD-VGG16
DATASETS=(b-59-850 SEILS Mus-Tradicional/c-combined Mus-Tradicional/m-combined)
N_IMAGES=(1 2 3 4 5 6 7 8 16 32 64)
FOLD=0 # Fixed since experiments are always on the fold 0

for MODEL in ${MODELS[*]}
do
  for DATASET in ${DATASETS[*]}
  do
    for IMAGES in ${N_IMAGES[*]}
    do
        # echo ""
        # for FOLD in 0 #1 2 3 4
        # do
        for split in val test
        do
            # echo "" 
            echo "python -u test.py ${MODEL} ${DATASET} --n_images=${IMAGES} --split=${split} ${FOLD}"
            python -u test.py ${MODEL} ${DATASET} --n_images=${IMAGES} --split=${split} ${FOLD}
            fold=$(($((fold))+$((1))))
            fold=$(($((fold))%$((5))))
            # echo ""
        done
        echo ""
        #done
    done
  done
done

