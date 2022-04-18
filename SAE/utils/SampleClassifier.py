from __future__ import print_function, division
import scipy


import tensorflow as tf

import progressbar

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Convolution2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import shuffle
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import cv2

from sklearn.utils import class_weight


try:
    from DataLoader import *
except:
    from utils.DataLoader import *

try:
    from Fscore import *
except:
    from utils.Fscore import *

try:
    from ModelConfig import *
except:
    from utils.ModelConfig import *

    
class SampleClassifier():
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
                    pathfile_saved_model, 
                    pathfile_results,
                    pathdir_training_images,
                    pathdir_testing_images,
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
        self.pathfile_results = pathfile_results
        self.pathdir_training_images = pathdir_training_images
        self.pathdir_testing_images = pathdir_testing_images
        self.max_counts_not_improving = 10
        self.counts_not_improving = 0

        self.with_early_stop = True

        optimizer = Adam()


        # Build and compile the model
        self.model = self.build_model()
        self.model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        

        

    def build_model(self):
        """CLF"""

        input = Input(shape=self.img_shape)
        x = input
        
        for _ in range(self.num_blocks):
            x = Convolution2D(filters=self.n_filters,kernel_size=self.kernel_shape, padding='same')(x)
            if (self.with_batch_normalization):
                x = BatchNormalization(axis=self.bn_axis)(x)

            #x = LeakyReLU(alpha=0.0001)(x)
            x = Activation(self.activation_function)(x)
            if (self.pool > 1):
                x = MaxPooling2D(pool_size=(self.pool, self.pool))(x)
       
        # Prediction
        x = Flatten()(x)
        x = Dense(self.num_classes)(x)
        x = Activation('softmax')(x)

        model = Model(inputs=input, outputs=x)
        model.summary()
        return model 
    

    def train(
                self,
                X_train, Y_train,
                X_val, Y_val,
                model_config,
                seed_train = 5,
                seed_val = 11):
        assert(isinstance(model_config, ModelConfig))
        self.best_fscore_val = FScoreValues(fscore=0., precision=0., recall=0.)
        self.best_acc = 0.
        self.counts_not_improving = 0
        self.best_epoch = 0

        save_model = True

        number_train_samples = len(X_train)

        z = list(zip(X_train, Y_train))

        X_train, Y_train = shuffle(X_train, Y_train, random_state=seed_train)
        X_val, Y_val = shuffle(X_val, Y_val, random_state=seed_val)

        np.random.seed(5)
        np.random.shuffle(z)
        X_train[:], Y_train[:] = zip(*z)

        Y_train_argmax = np.argmax(Y_train, axis=1)
        classes = np.array(list(range(self.num_classes)))

        Y_train_argmax_with_1_sample_min_per_class = np.concatenate((Y_train_argmax, range(0, self.num_classes)))
        shape_weights = class_weight.compute_class_weight('balanced', classes, Y_train_argmax_with_1_sample_min_per_class)

        mean_train_imgs = np.mean(X_train)
        std_train_imgs = np.std(X_train)

        X_train = model_config.applyNormalizationDataset(imgs=X_train, mean = mean_train_imgs, std = std_train_imgs)
        X_val = model_config.applyNormalizationDataset(imgs=X_val, mean = mean_train_imgs, std = std_train_imgs)

        f = open(self.pathfile_results,'w+')

        for epoch in range(1, model_config.getNumberEpochs()+1):

            idx_start_batch = 0
            idx_end_batch = model_config.getBatchSize()

            progress_bar = progressbar.ProgressBar(maxval=number_train_samples, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            print ("Ep. %d (%d samples)" % (epoch, number_train_samples))
            progress_bar.start()

            while (idx_start_batch < idx_end_batch):

                imgs_train = X_train[idx_start_batch:idx_end_batch]
                labels_train = Y_train[idx_start_batch:idx_end_batch]

                d_loss_train = self.model.train_on_batch(x=imgs_train, y=labels_train, class_weight=shape_weights)

                idx_start_batch = idx_end_batch
                idx_end_batch = (idx_end_batch + model_config.getBatchSize()) % number_train_samples

                progress_bar.update(idx_start_batch)

            progress_bar.finish()

            

            pred_val_target = self.model.predict(X_val)
            pred_val_target = np.argmax(pred_val_target, axis=1)
            Y_val_argmax = np.argmax(Y_val, axis=1)
            

            acc = accuracy(pred_val_target, Y_val_argmax)
            macro_fscore_val = macroFScore(pred_val_target, Y_val_argmax)
            macro_precision_val = macroPrecision(pred_val_target, Y_val_argmax)
            macro_recall_val = macroRecall(pred_val_target, Y_val_argmax)
            fscore_values_val = FScoreValues(fscore=macro_fscore_val, precision=macro_precision_val, recall=macro_recall_val)

            f.write(str(fscore_values_val) + "\n")

            X_train, Y_train = shuffle(X_train, Y_train, random_state=epoch)

            #if self.best_fscore_val.isImprovedBy(fscore_values_val):
            if acc > self.best_acc:
                print ("Model improved: F1:%s ---> %s; ACC: %.4f ---> %.4f" %(self.best_fscore_val, fscore_values_val, self.best_acc, acc))
                if (self.counts_not_improving <= self.max_counts_not_improving):
                    if save_model:
                        self.model.save(self.pathfile_saved_model)
                        print("Classifier model saved in " + self.pathfile_saved_model)
                else:
                    print ("Model improved but not saved for early stop: F1:%s ---> %s (acc: %.4f ---> %.4f)" %(str(self.best_fscore_val), str(fscore_values_val), self.best_acc, acc))
                    save_model = False

                self.best_fscore_val = fscore_values_val
                self.best_acc = acc
                self.counts_not_improving = 0
                self.best_epoch = epoch
            else:
                self.counts_not_improving = self.counts_not_improving + 1
                print ("mFscore not improved (%d/%d)(F1: %s --> %s) ; (ACC: %.4f --> %.4f). Best epoch: %d" % \
                                        (self.counts_not_improving, 
                                        self.max_counts_not_improving, 
                                        str(self.best_fscore_val), str(fscore_values_val),
                                        self.best_acc, acc, 
                                        self.best_epoch))
                if (self.with_early_stop and self.counts_not_improving >= self.max_counts_not_improving):
                    print ("Early stop.")
                    f.close()
                    return
            

        f.close()
        
    
    def test(self, X_test, Y_test, mean_train_imgs, std_train_imgs, model_config):
        assert(isinstance(model_config, ModelConfig))

        print("NUM TEST SAMPLES: " + str(len(X_test)))
        X_test_norm = model_config.applyNormalizationDataset(imgs=X_test, mean = mean_train_imgs, std = std_train_imgs)
        Y_test_argmax = np.argmax(Y_test, axis=1)
        
        symbol_prediction = self.model.predict(x=X_test_norm, batch_size=model_config.getBatchSize(), verbose=1)
            
        symbol_prediction_argmax = np.argmax(symbol_prediction, axis=1)
        macro_fscore = macroFScore(symbol_prediction_argmax, Y_test_argmax)
        macro_precision = macroPrecision(symbol_prediction_argmax, Y_test_argmax)
        macro_recall = macroRecall(symbol_prediction_argmax, Y_test_argmax)
        fscore_values = FScoreValues(fscore=macro_fscore, precision=macro_precision, recall=macro_recall)


        if (len(X_test) == 1):
            print ("GT:" + "\t" + str(Y_test_argmax) + "\tPRED:\t" + str(symbol_prediction_argmax))
        else:
            print ('-'*40 + "GROUND TRUTH" + '-'*40)
            print (Y_test_argmax)
            print ('-'*40 + "PREDICTION" + '-'*40)
            print (symbol_prediction_argmax)
            
            acc = accuracy(symbol_prediction_argmax, Y_test_argmax)
            print ("Fscore: " + str(fscore_values))
            print ("ACC: %.4f" % (acc))


    def load_model(self):
        self.model = load_model(self.pathfile_saved_model)


    def predict(self, x, batch_size, verbose):
        return self.model.predict(x=x, batch_size=batch_size, verbose=verbose)


    def sample_images(self, imgs, labels, binarized_imgs, epoch = None):
        r, c = 6, 8

        binarized_img3D = np.zeros((16, self.img_rows, self.img_cols, self.channels))
        labels_target_img3D = np.zeros((16, self.img_rows, self.img_cols, self.channels))
        idx = 0
        for channel in range(self.channels):
            binarized_img3D[:,:,:,channel] = binarized_imgs[0:16]
            labels_target_img3D[:,:,:,channel] = labels[0:16,:,:,1]
            idx = idx + 1


        gen_imgs = np.concatenate([imgs[0:16], labels_target_img3D[0:16], binarized_img3D[0:16]])
        gen_imgs = (1-gen_imgs) * 255
        gen_imgs = gen_imgs.astype(np.uint8)

        #titles = ['Original', 'Translated']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                im = gen_imgs[cnt,:,:,:]
                axs[i,j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                #axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        if (epoch is None):
            fig.savefig("images/CLF.png")
        else:
            fig.savefig("images/%d_CLF.png" % (epoch))
        plt.close()




