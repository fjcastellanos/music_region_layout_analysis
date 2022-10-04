MODELS=Faster-RCNN #(Faster-RCNN ResNet MobileNet YOLO)
N_PAGES=1 #(1 2 4 8 16 32)
N_IMAGES=1
DATASETS=b-59-850 #(b-59-850 SEILS Mus-Tradicional)
fold=0
epochs=1000
patience=30
# [0, 1, 2] Training # [3] Validation # [4] Test

for MODEL in ${MODELS[*]}
do
  for DATASET in ${DATASETS[*]}
  do
    for PAGES in ${N_PAGES[*]}
    do
        echo ""
        for FOLD in 0 1 2 3 4
        do
            echo "" 
            echo "python -u train.py ${MODEL} ${DATASET} --n_pages=${PAGES} --n_images=${N_IMAGES} ${fold} $epochs $patience"
            python -u train.py ${MODEL} ${DATASET} --n_pages=${PAGES} --n_images=${N_IMAGES} ${fold} $epochs $patience
            fold=$(($((fold))+$((1))))
            fold=$(($((fold))%$((5))))
            echo ""
        done
    done
  done
done

