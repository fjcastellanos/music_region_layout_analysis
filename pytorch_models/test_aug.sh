#!/bin/bash
# test with data augmentation

# Arguments: 
    # $1: (MobileNet Faster-RCNN RetinaNet SSD-VGG16 SAE)
    # $2: (b-59-850 SEILS Mus-Tradicional/c-combined Mus-Tradicional/c-combined)
    # $3: (0 1)

#MODELS=MobileNet RetinaNet Faster-RCNN SSD-VGG16 #RetinaNet #SSD-VGG16 #MobileNet #(Faster-RCNN ResNet MobileNet YOLO)
MODELS=$1 #(Faster-RCNN MobileNet RetinaNet SSD-VGG16)
N_PAGES=(1 2 3 4 5 6 7 8 16 32 64)
N_IMAGES=100 #(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
ROTATIONS=(0 1)
DATASETS=$2 #(b-59-850 SEILS Mus-Tradicional/c-combined Mus-Tradicional/m-combined) #$2 #SEILS #Mus-Tradicional/m-combined #(b-59-850 SEILS Mus-Tradicional)
FOLD=0

for MODEL in ${MODELS[*]}
do
    for DATASET in ${DATASETS[*]}
    do
        for ROTATION in ${ROTATIONS[*]}
        do
            for PAGES in ${N_PAGES[*]}
            do
                for IMAGES in ${N_IMAGES[*]}
                do
                    # echo ""
                    # for FOLD in 0 # 1 2 3 4
                    # do
                    for split in test val #val test #train val test
                    do
                        # echo "" 
                        echo "python -u test.py ${MODEL} ${DATASET} --n_pages=${PAGES} --n_images=${IMAGES} --unif_rotation=${ROTATION} --split=${split} ${FOLD}"
                        python -u test.py ${MODEL} ${DATASET} --n_pages=${PAGES} --n_images=${IMAGES} --unif_rotation=${ROTATION} --split=${split} ${FOLD}
                        fold=$(($((fold))+$((1))))
                        fold=$(($((fold))%$((5))))
                        # echo ""
                    done
                    echo ""
                    # done
                done
            done
        done
    done
done
