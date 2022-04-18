type="random-auto"
NumberNewImages=100
options="--uniform_rotate"

dataset="Mus-Tradicional/c-combined"

for fold in 0 1 2 3 4; do 
    for pages in 1 2 4 8 16 32 64; do 
        pathjson="datasets/JSON/Folds/${dataset}/fold${fold}/train.txt"

        json_dataset_serial=${dataset// /.}
        json_dataset_serial=${json_dataset_serial////-}

        log_file="logs/fold${fold}_${type}_${json_dataset_serial}_${NumberNewImages}newpages_${pages}originalpages_${options}.txt"
        echo $pathjson
        echo $log_file
        python3 -u ./main.py \
                    -type ${type} \
                    -n ${NumberNewImages} \
                    -txt_train ${pathjson} \
                    -pages ${pages} \
                    ${options} > ${log_file}
    done
done


dataset="Mus-Tradicional/m-combined"

for fold in 0 1 2 3 4; do 
    for pages in 1 2 4 8 16 32 64; do 
        pathjson="datasets/JSON/Folds/${dataset}/fold${fold}/train.txt"

        json_dataset_serial=${dataset// /.}
        json_dataset_serial=${json_dataset_serial////-}

        log_file="logs/fold${fold}_${type}_${json_dataset_serial}_${NumberNewImages}newpages_${pages}originalpages_${options}.txt"
        echo $pathjson
        echo $log_file
        python3 -u ./main.py \
                    -type ${type} \
                    -n ${NumberNewImages} \
                    -txt_train ${pathjson} \
                    -pages ${pages} \
                    ${options} > ${log_file}
    done
done

dataset="Mus-Tradicional/mision02"

for fold in 0 1 2 3 4; do 
    for pages in 1 2 4 8 16 32 64; do 
        pathjson="datasets/JSON/Folds/${dataset}/fold${fold}/train.txt"

        json_dataset_serial=${dataset// /.}
        json_dataset_serial=${json_dataset_serial////-}

        log_file="logs/fold${fold}_${type}_${json_dataset_serial}_${NumberNewImages}newpages_${pages}originalpages_${options}.txt"
        echo $pathjson
        echo $log_file
        python3 -u ./main.py \
                    -type ${type} \
                    -n ${NumberNewImages} \
                    -txt_train ${pathjson} \
                    -pages ${pages} \
                    ${options} > ${log_file}
    done
done

dataset="SEILS"

for fold in 0 1 2 3 4; do 
    for pages in 1 2 4 8 16 32 64; do 
        pathjson="datasets/JSON/Folds/${dataset}/fold${fold}/train.txt"

        json_dataset_serial=${dataset// /.}
        json_dataset_serial=${json_dataset_serial////-}

        log_file="logs/fold${fold}_${type}_${json_dataset_serial}_${NumberNewImages}newpages_${pages}originalpages_${options}.txt"
        echo $pathjson
        echo $log_file
        python3 -u ./main.py \
                    -type ${type} \
                    -n ${NumberNewImages} \
                    -txt_train ${pathjson} \
                    -pages ${pages} \
                    ${options} > ${log_file}
    done
done

dataset="b-59-850"

for fold in 0 1 2 3 4; do 
    for pages in 1 2 4 8 16 32 64; do 
        pathjson="datasets/JSON/Folds/${dataset}/fold${fold}/train.txt"

        json_dataset_serial=${dataset// /.}
        json_dataset_serial=${json_dataset_serial////-}

        log_file="logs/fold${fold}_${type}_${json_dataset_serial}_${NumberNewImages}newpages_${pages}originalpages_${options}.txt"
        echo $pathjson
        echo $log_file
        python3 -u ./main.py \
                    -type ${type} \
                    -n ${NumberNewImages} \
                    -txt_train ${pathjson} \
                    -pages ${pages} \
                    ${options} > ${log_file}
    done
done
