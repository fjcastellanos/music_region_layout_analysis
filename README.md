# Region-based Layout Analysis of Music Score Images
This repository corresponds to the paper: **Region-based Layout Analysis of Music Score Images**, submitted to the journal 'Expert Systems to Applications'.

## BibTex Reference

```
@article{2022_Castellanos_regions,
  author    = {Francisco J. Castellanos and
               Carlos Garrido{-}Munoz and
               Antonio R{\'{\i}}os{-}Vila and
               Jorge Calvo{-}Zaragoza},
  title     = {Region-based Layout Analysis of Music Score Images},
  journal   = {CoRR},
  volume    = {abs/2201.04214},
  year      = {2022},
  eprinttype = {arXiv},
  eprint    = {2201.04214}
}
```

## SAE
The SAE folder includes the code that implements the Selectional Auto-encoder. This program has a series of parameters, as described below:

  **--mode** : Execution mode (train or test).\
  **--gpu** : Identifier of GPU.\
  **--cmode** : Color mode (0 for grayscale, 1 for RGB).\
  **--s-width** : Width size of the rescaled image.\
  **--s-height** : Height size of the rescaled image.\
  **--k-height** : Kernel height.\
  **--k-width** : Kernel width.\
  **--nfilt** : Number of filters to configure the convolutional layers.\
  **--batch** : Batch size.\
  **--norm** : Type of image normalization.\
  **--epochs** : Maximum number of epochs.\
  **--nbl** : Number of blocks in the encoder and decoder of the SAE model.\
  **--img** : It activates the mode for saving images to check the evolution of the training process.\
  **--graph** : It activates the mode to save the model graph.\
  **--post** : It activates a post-processing filter for improving the recognition.\
  **--th** : IoU threshold to compute metrics.\
  **--nimgs** : Number of images considered from the training set.\
  **--red** : Reduction factor to reduce vertically the regions before training.\
  **--labels** : Name of the labels to be used for training. This code uses the data configuration provided by the [MuReT tool](https://muret.dlsi.ua.es/muret/#/).

Example of use:

```[python]
python -u main.py ${model}
      --db-train dataset/train.txt
      --db-val dataset/val.txt
      --db-test dataset/test.txt
      --mode train
      --gpu 0
      --cmode 1
      --s-width 512
      --s-height 512
      --k-height 3
      --k-width 3
      --nfilt 128
      --batch 16
      --norm inv255
      --epochs 300
      --nbl 3
      --nimgs 32
      --labels staff
      --labels lyrics
      --th 0.55
      --red 0.2
```

## Data augmentation
This folder contains the code for generating the augmented images used in the paper. There is an example of use in the script:

Parameters of this code:

  **-type** : It configures the manner of generating data. The used in the paper is "random-auto".\
  **-n** : Number of new semi-synthetic images.\
  **-txt_train** : Path to the folder that contains the json files with the ground-truth data. This Ground-truth data has been generated through the [MuReT tool](https://muret.dlsi.ua.es/muret/#/).\
  **-pages** : Number of pages to be considered as real available pages.\
  **--uniform_rotate** : It activates the mode for keeping a uniform rotation for all regions within the same page.


```
data_aug/generate_daug_all.sh
```

Another example of use:
```
python3 -u ./main.py \
      -type ${type} \
      -n 100 \
      -txt_train dataset/json_files.json \
      -pages 10 \
      --uniform_rotate
```


## End-to-end
The code for the end-to-end approach used in this work can be found in [end-to-end code](https://github.com/HISPAMUS/end-to-end-recognition/tree/develop/code)


