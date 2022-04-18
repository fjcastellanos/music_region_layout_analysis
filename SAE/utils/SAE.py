# Binarizer model based on Selectional Auto-Encoder

from __future__ import print_function, division
import scipy


import tensorflow as tf

from sklearn.utils import shuffle
import progressbar

#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Convolution2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam,SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import cv2


try:
    from ModelConfig import ModelConfig
    from Fscore import *
    from FscoreRegion import *
    from DataLoader import *
    from file_manager import FileManager
    from ImageProcessing import *
    from image_manager import *
    from RegionExtraction import *
    from GTJSONReaderMuret import *
except:
    from utils.Fscore import *
    from utils.FscoreRegion import *
    from utils.ModelConfig import ModelConfig
    from utils.DataLoader import *
    from utils.file_manager import FileManager
    from utils.ImageProcessing import *
    from utils.image_manager import *
    from utils.RegionExtraction import *
    from utils.GTJSONReaderMuret import *


    
class SAE():
    def __init__(
                    self,
                    input_shape, 
                    kernel_shape, 
                    n_filters, 
                    pool, 
                    bn_axis, 
                    activation_function, 
                    with_batch_normalization,
                    num_blocks,
                    with_saving_images,
                    pathfile_saved_model,
                    num_classes = 2):
        # Input shape
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.channels = input_shape[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.kernel_shape = kernel_shape
        self.n_filters = n_filters
        self.pool = pool
        self.bn_axis = bn_axis
        self.activation_function = activation_function
        self.with_batch_normalization = with_batch_normalization
        self.num_blocks = num_blocks
        self.pathfile_saved_model = pathfile_saved_model
        self.max_counts_not_improving = 5
        self.counts_not_improving = 0
        self.with_early_stop = False
        self.with_saving_images = with_saving_images

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        #optimizer = Adam(0.0001, 0.5)
        optimizer = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)


        # Build and compile the model
        self.model = self.build_model()
        self.model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['mse', 'accuracy'])

        

    def build_model(self):
        """SAE"""

        input = Input(shape=self.img_shape)
        encoder = Conv2D(filters=self.n_filters,kernel_size=self.kernel_shape, padding='same', input_shape=self.img_shape)(input)

        if (self.with_batch_normalization):
            encoder = BatchNormalization(axis=self.bn_axis)(encoder)
        
        #encoder = LeakyReLU(alpha=0.0001)(encoder)
        encoder = Activation(self.activation_function)(encoder)
        encoder = MaxPooling2D(pool_size=(self.pool, self.pool))(encoder)

        for _ in range(self.num_blocks - 1):
            encoder = Convolution2D(filters=self.n_filters,kernel_size=self.kernel_shape, padding='same')(encoder)
            if (self.with_batch_normalization):
                encoder = BatchNormalization(axis=self.bn_axis)(encoder)

            #encoder = LeakyReLU(alpha=0.0001)(encoder)
            encoder = Activation(self.activation_function)(encoder)
            if (self.pool > 1):
                encoder = MaxPooling2D(pool_size=(self.pool, self.pool))(encoder)
            
        #latent_space = encoder

        # Decoding

        decoder = encoder
        for _ in range(self.num_blocks):
            decoder = Convolution2D(filters=self.n_filters,kernel_size=self.kernel_shape, padding='same')(decoder)
            if (self.with_batch_normalization):
                decoder = BatchNormalization(axis=self.bn_axis)(decoder)
            #decoder = LeakyReLU(alpha=0.0001)(decoder)
            decoder = Activation(self.activation_function)(decoder)
            if (self.pool > 1):
                decoder = UpSampling2D((self.pool, self.pool))(decoder)

     
        # Prediction
        decoder = Convolution2D(self.num_classes, kernel_size=self.kernel_shape, padding='same')(decoder)
        decoder = Activation('sigmoid')(decoder)

        model = Model(inputs=input, outputs=decoder)
        #model.summary()
        return model


    def train(
                self,
                list_json_files_train,
                list_json_files_val,
                generator_train,
                generator_val,
                model_config):

        self.best_macro_fscore = 0.
        self.counts_not_improving = 0
        self.best_epoch = 0

        assert(isinstance(model_config, ModelConfig))
        
        number_source_samples_train = len(list_json_files_train)
        number_source_samples_val = len(list_json_files_val)
  
        seed_train = 5
        seed_val = 11

        epochs = model_config.getNumberEpochs()
        output_path = self.pathfile_saved_model
        batch_size = model_config.getBatchSize()
        steps_per_epoch_val = np.ceil(number_source_samples_val / batch_size)
        steps_per_epoch_train = np.ceil(number_source_samples_train / batch_size)

        callbacks_list = [
                    ModelCheckpoint(
                        output_path,
                        save_best_only=True,
                        monitor="val_accuracy",
                        verbose=1,
                        mode="max"
                    ),
                    EarlyStopping(monitor="val_accuracy", patience=30, verbose=0, mode="max"),
        ]

        self.model.fit(
                    generator_train,
                    verbose=2,
                    steps_per_epoch=steps_per_epoch_train,
                    validation_data=generator_val,
                    validation_steps=steps_per_epoch_val,
                    callbacks=callbacks_list,
                    epochs=epochs)



    def load_model(self):
        self.model.load_weights(self.pathfile_saved_model)




