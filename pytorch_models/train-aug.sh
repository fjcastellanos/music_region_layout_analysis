#!/bin/bash
# train with data augmentation

MODELS=Faster-RCNN #(Faster-RCNN ResNet MobileNet YOLO)
N_PAGES=(1 2 3 4 5 6 7 8 16 32 64)
N_IMAGES=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
ROTATIONS=1 #(0 1)
DATASETS=b-59-850 #(b-59-850 SEILS Mus-Tradicional)
EPOCHS=300
PATIENCE=30

for MODEL in ${MODELS} #${MODELS[*]}
do
    for DATASET in ${DATASETS} #${MODELS[*]}
    do
        for ROTATION in ${ROTATIONS}
        do
            for PAGES in ${N_PAGES[*]}
            do
                for IMAGES in ${N_IMAGES[*]}
                do
                    # echo ""
                    for FOLD in 0 # 1 2 3 4
                    do
                        # echo "" 
                        echo "python -u train-torch.py ${MODEL} ${DATASET} --n_pages=${PAGES} --n_images=${IMAGES} --unif_rotation=${ROTATION} ${FOLD} $EPOCHS $PATIENCE"
                        # python -u train-torch.py ${MODEL} ${DATASET} --n_pages=${PAGES} --n_images=${IMAGES} --unif_rotation=${ROTATION} ${FOLD} $EPOCHS $PATIENCE
                        fold=$(($((fold))+$((1))))
                        fold=$(($((fold))%$((5))))
                        # echo ""
                    done
                done
            done
        done
    done
done
