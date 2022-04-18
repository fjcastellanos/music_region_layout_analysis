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
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import cv2

from sklearn.utils import class_weight


try:
    from DataLoader import *
    from Fscore import *
    from ModelConfig import *
    from SampleClassifier import *
except:
    from utils.DataLoader import *
    from utils.Fscore import *
    from utils.ModelConfig import *
    from utils.SampleClassifier import *


    
class DomainClassifier():
    def __init__(
                    self,
                    domain_classifier,
                    list_classifiers_per_domain,
                    with_assemble,
                    considered_classes,
                    considered_collections,
                    considered_domains):
        self.list_classifiers_per_domain = list_classifiers_per_domain
        self.domain_classifier = domain_classifier
        self.with_assemble = with_assemble
        self.considered_classes = considered_classes
        self.considered_collections = considered_collections
        self.considered_domains = considered_domains

           
    def getPredictionsByDomain(self, idx_domain_predictions, X_test_norm, Y_test_symbol_argmax, model_config):

        count = 0

        expected = []
        prediction = []
        for idx_domain_prediction_i in idx_domain_predictions:

            symbol_clf = self.list_classifiers_per_domain[idx_domain_prediction_i]
            expected_i = Y_test_symbol_argmax[count]
            prediction_i = symbol_clf.predict(x=np.array([X_test_norm[count]]), batch_size=model_config.getBatchSize(), verbose=0)
            prediction_i = np.argmax(prediction_i, axis=1)

            expected.append(expected_i)
            prediction.append(prediction_i)
            
            count = count + 1

        return expected, prediction

    def getPredictionsByDomainWithAssemble(
                            self, 
                            domain_predictions, 
                            X_test_norm, 
                            Y_test_symbol_argmax, 
                            model_config):
        count = 0

        expected = []
        prediction = []

        assert(len(X_test_norm) == len(Y_test_symbol_argmax) == len(domain_predictions))
        assert(len(self.list_classifiers_per_domain) == domain_predictions.shape[1])

        pondered_predictions = np.zeros((len(domain_predictions), len(self.considered_classes)))
        for idx_test in range(len(Y_test_symbol_argmax)):

            idx_clf = 0
            expected_i = Y_test_symbol_argmax[idx_test]

            for symbol_clf in self.list_classifiers_per_domain:
                prediction_clf = symbol_clf.predict(x=np.array([X_test_norm[idx_test]]), batch_size=model_config.getBatchSize(), verbose=0)                
                prediction_cld_pondered = domain_predictions[idx_test][idx_clf] * prediction_clf

                pondered_predictions[idx_test,:] = pondered_predictions[idx_test,:] + prediction_cld_pondered
                
                idx_clf = idx_clf + 1

            
            expected.append(expected_i)
            
            count = count + 1

        pondered_predictions_argmax = np.argmax(pondered_predictions, axis=1)

        return expected, pondered_predictions_argmax


    def test(self, X_test, Y_test_domain, Y_test_symbol, mean_train_imgs, std_train_imgs, model_config):
        assert(isinstance(model_config, ModelConfig))

        X_test_norm = model_config.applyNormalizationDataset(imgs=X_test, mean = mean_train_imgs, std = std_train_imgs)
        Y_test_symbol_argmax = np.argmax(Y_test_symbol, axis=1)

        domain_predictions = self.domain_classifier.predict(x=X_test_norm, batch_size=model_config.getBatchSize(), verbose=1)
        
        domain_confusion_matrix = np.tile(0, (len(self.considered_collections), len(self.considered_domains)))

        idx_domain_predictions = np.argmax(domain_predictions, axis=1)
        idx_test_domain_expecteds = np.argmax(Y_test_domain, axis=1)

        print ("Computing the domain confusion matrix")
        idx = 0
        for idx_domain_prediction in idx_domain_predictions:
            domain_confusion_matrix[idx_test_domain_expecteds[idx], idx_domain_prediction] = domain_confusion_matrix[idx_test_domain_expecteds[idx], idx_domain_prediction] + 1
            idx = idx + 1

        if (self.with_assemble):
            expected, prediction = self.getPredictionsByDomainWithAssemble(
                                        domain_predictions=domain_predictions, 
                                        X_test_norm=X_test_norm, 
                                        Y_test_symbol_argmax=Y_test_symbol_argmax, 
                                        model_config=model_config)
        else:
            expected, prediction = self.getPredictionsByDomain(
                                        idx_domain_predictions=idx_domain_predictions, 
                                        X_test_norm=X_test_norm, 
                                        Y_test_symbol_argmax=Y_test_symbol_argmax, 
                                        model_config=model_config)

        
        expected_arr = np.array(expected)
        prediction_arr = np.array(prediction)

        macro_fscore = macroFScore(prediction_arr, expected_arr)
        macro_precision = macroPrecision(prediction_arr, expected_arr)
        macro_recall = macroRecall(prediction_arr, expected_arr)
        fscore_values = FScoreValues(fscore=macro_fscore, precision=macro_precision, recall=macro_recall)
        #fscore_eachclass = FScoreAllClasses(prediction_arr, expected_arr)
        acc = accuracy(prediction_arr, expected_arr)
        print('-'*80)
        print ("Test result:")
        print ("F1:" + str(fscore_values) + "\tAcc: %.4f" % acc)
        #print ("F1 for each class:")
        #print (fscore_eachclass)
        print ("ALL COLLECTIONS: " + str(self.considered_collections))
        print("DOMAINS IN TRAINING: " + str(self.considered_domains))
        print ("****************Confusion Matrix***************")
        print (domain_confusion_matrix)
        print ("***********************************************")



if __name__ == '__main__':
    
    pathdir_exec = os.path.dirname(os.path.abspath(__file__))
    pathdir_gt_files_source = pathdir_exec + "/" + "databases/datasets_bin/GT/Einsiedeln"
    pathdir_src_files_source = pathdir_exec + "/" + "databases/datasets_bin/SRC/Einsiedeln"

    pathdir_gt_files_target = pathdir_exec + "/" + "databases/datasets_bin/GT/Salzinnes"
    pathdir_src_files_target = pathdir_exec + "/" + "databases/datasets_bin/SRC/Salzinnes"

    