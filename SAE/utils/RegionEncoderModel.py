# Binarizer model based on Selectional Auto-Encoder

from __future__ import print_function, division
import scipy


import tensorflow as tf

from sklearn.utils import shuffle
import progressbar

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
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
    from DataLoader import *
    from file_manager import FileManager
    from ImageProcessing import *
    from RegionExtraction import *
except:
    from utils.Fscore import *
    from utils.ModelConfig import ModelConfig
    from utils.DataLoader import *
    from utils.file_manager import FileManager
    from utils.ImageProcessing import *
    from utils.RegionExtraction import *


    
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
        self.pathdir_training_images=pathdir_training_images
        self.pathdir_testing_images=pathdir_testing_images
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
        model.summary()
        return model


    def train(
                self,
                X_train_source, Y_train_source,
                X_train_target, Y_train_target,
                X_val_source, Y_val_source,
                X_val_target, Y_val_target,
                model_config):

        self.best_macro_fscore = 0.
        self.counts_not_improving = 0
        self.best_epoch = 0

        assert(isinstance(model_config, ModelConfig))
        
        mean_training_imgs = np.mean(X_train_source)
        std_training_imgs = np.std(X_train_source)
        number_source_samples = len(X_train_source)

        pathfile_logs = self.pathfile_results.replace("results", "logs").replace(".txt", "")


        FileManager.deleteFolder(pathfile_logs)
          
        tensorboard = TensorBoard(
                log_dir=pathfile_logs,
                histogram_freq=0,
                batch_size=model_config.getBatchSize(),
                write_graph=True,
                write_grads=True
                )
        tensorboard.set_model(self.model)

        # Transform train_on_batch return value
        # to dict expected by on_batch_end callback
        def named_logs(
                    fscore_source_val, 
                    fscore_target_val,
                    fscore_target_val_class0, fscore_target_val_class1, 
                    fscore_source_val_class0, fscore_source_val_class1):
            result = {}
            result["F1_VAL_source"] = fscore_source_val
            result["F1_VAL_target"] = fscore_target_val
            result["F1_VAL_source_class0"] = fscore_source_val_class0
            result["F1_VAL_source_class1"] = fscore_source_val_class1
            result["F1_VAL_target_class0"] = fscore_target_val_class0
            result["F1_VAL_target_class1"] = fscore_target_val_class1
            
            return result
        

        seed_train = 5
        seed_val = 11

        X_train_source = model_config.applyNormalizationDataset(imgs=X_train_source, mean = mean_training_imgs, std = std_training_imgs)
        Y_train_source = (Y_train_source<128)
        Y_train_source = to_categorical(Y_train_source)

        X_train_target = model_config.applyNormalizationDataset(imgs=X_train_target, mean = mean_training_imgs, std = std_training_imgs)
        Y_train_target = (Y_train_target<128)
        Y_train_target = to_categorical(Y_train_target)

        X_val_source = model_config.applyNormalizationDataset(imgs=X_val_source, mean = mean_training_imgs, std = std_training_imgs)
        Y_val_source = (Y_val_source<128)
        Y_val_source = to_categorical(Y_val_source)

        X_val_target = model_config.applyNormalizationDataset(imgs=X_val_target, mean = mean_training_imgs, std = std_training_imgs)
        Y_val_target = (Y_val_target<128)
        Y_val_target = to_categorical(Y_val_target)

        X_train_source, Y_train_source = shuffle(X_train_source, Y_train_source, random_state=seed_train)
        X_train_target, Y_train_target = shuffle(X_train_target, Y_train_target, random_state=seed_train)
        X_val_source, Y_val_source = shuffle(X_val_source, Y_val_source, random_state=seed_val)
        X_val_target, Y_val_target = shuffle(X_val_target, Y_val_target, random_state=seed_val)

        f_val_target = open(self.pathfile_results + "_val_target",'w+')
        f_val_source = open(self.pathfile_results + "_val_source",'w+')
        
        for epoch in range(1,model_config.getNumberEpochs()+1):
            idx_start_batch = 0
            idx_end_batch = model_config.getBatchSize()

            progress_bar = progressbar.ProgressBar(maxval=number_source_samples, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            print ("Ep. %d (%d samples)" % (epoch, number_source_samples))
            progress_bar.start()

            while (idx_start_batch < idx_end_batch):

                imgs_train = X_train_source[idx_start_batch:idx_end_batch]
                labels_train = Y_train_source[idx_start_batch:idx_end_batch]

                d_loss_train = self.model.train_on_batch(imgs_train, labels_train)

                idx_start_batch = idx_end_batch
                idx_end_batch = (idx_end_batch + model_config.getBatchSize()) % number_source_samples

                progress_bar.update(idx_start_batch)

            progress_bar.finish()

            
            imgs_pred_val_source = self.model.predict(x=X_val_source, batch_size=model_config.getBatchSize(), verbose=0)
            imgs_pred_val_target = self.model.predict(x=X_val_target, batch_size=model_config.getBatchSize(), verbose=0)

            evaluation = self.model.evaluate(x=X_val_target, y=Y_val_target, batch_size=model_config.getBatchSize(), verbose=0)

            '''
            [binarized_imgs_1D_val_source, labels_cat_1D_val_source] = prepare_imgs_to_sklearn(imgs_pred_val_source, Y_val_source)
            macro_fscore_val_source = macroFScore(binarized_imgs_1D_val_source, labels_cat_1D_val_source)
            macro_fscore_val_source_classes = FScoreAllClasses(binarized_imgs_1D_val_source, labels_cat_1D_val_source, self.num_classes)
            assert(len(macro_fscore_val_source_classes) == self.num_classes)
            macro_precision_val_source = 0#macroPrecision(binarized_imgs_1D_val_source, labels_cat_1D_val_source)
            macro_recall_val_source = 0#macroRecall(binarized_imgs_1D_val_source, labels_cat_1D_val_source)
            fscore_values_val_source = FScoreValues(fscore=macro_fscore_val_source, precision=macro_precision_val_source, recall=macro_recall_val_source)
            '''

            [binarized_imgs_1D_val_target, labels_target_cat_1D_val_target] = prepare_imgs_to_sklearn(imgs_pred_val_target, Y_val_target)
            macro_fscore_val_target = macroFScore(binarized_imgs_1D_val_target, labels_target_cat_1D_val_target)
            macro_fscore_val_target_classes = FScoreAllClasses(binarized_imgs_1D_val_target, labels_target_cat_1D_val_target, self.num_classes)
            assert(len(macro_fscore_val_target_classes) == self.num_classes)
            print("Fscore in classes: " + str(macro_fscore_val_target_classes))
            macro_precision_val_target = 0#macroPrecision(binarized_imgs_1D_val_target, labels_target_cat_1D_val_target)
            macro_recall_val_target = 0#macroRecall(binarized_imgs_1D_val_target, labels_target_cat_1D_val_target)
            fscore_values_val_target = FScoreValues(fscore=macro_fscore_val_target, precision=macro_precision_val_target, recall=macro_recall_val_target)

            s = "%d : mF1: %.4f( - mP: %.4f( - mR: %.4f() - " %  \
                            (epoch, 
                            macro_fscore_val_target, 
                            macro_precision_val_target,
                            macro_recall_val_target)
                            
            for idx_metric in range(len(self.model.metrics_names)):
                metric_name = self.model.metrics_names[idx_metric]
                result = evaluation[idx_metric]
                s = s + "(%s-%.4f)" %(metric_name, result)
            print(s)

            if self.with_saving_images:
                Y_val_target_argmax_16 = np.argmax(Y_val_target[0:16], axis=3)
                imgs_pred_val_target_argmax_16 = np.argmax(imgs_pred_val_target[0:16], axis=3)
                self.sample_images(is_testing=False, imgs=np.copy(X_val_target[0:16]), labels=Y_val_target_argmax_16, binarized_imgs=imgs_pred_val_target_argmax_16, model_config=model_config, epoch=epoch)

            
            tensorboard.on_epoch_end(epoch, named_logs(
                    fscore_source_val=0., 
                    fscore_target_val=macro_fscore_val_target,
                    fscore_target_val_class0=macro_fscore_val_target_classes[0], fscore_target_val_class1=macro_fscore_val_target_classes[1], 
                    fscore_source_val_class0=0., fscore_source_val_class1=0.
                    ))

            f_val_target.write(str(fscore_values_val_target) + "\n")
            #f_val_source.write(str(fscore_values_val_source) + "\n")

            X_train_source, Y_train_source = shuffle(X_train_source, Y_train_source, random_state=epoch)
            X_train_target, Y_train_target = shuffle(X_train_target, Y_train_target, random_state=epoch)

            if macro_fscore_val_target > self.best_macro_fscore:
                self.model.save(self.pathfile_saved_model)
                print("Classifier model saved in " + self.pathfile_saved_model)
                print ("mFscore improved: %.4f ---> %.4f" %(self.best_macro_fscore, macro_fscore_val_target))

                #if (self.with_early_stop and self.counts_not_improving >= self.max_counts_not_improving):
                #   self.model.save(self.pathfile_saved_model)
                #   print("Classifier model saved in " + self.pathfile_saved_model)
                #   print ("mFscore improved: %.4f ---> %.4f" %(self.best_macro_fscore, macro_fscore))
                #else:
                #   print ("mFscore improved (model not saved for early stop): %.4f ---> %.4f" %(self.best_macro_fscore, macro_fscore))

                self.best_macro_fscore = macro_fscore_val_target
                self.counts_not_improving = 0
                self.best_epoch = epoch
            else:
                self.counts_not_improving = self.counts_not_improving + 1
                print ("mFscore not improved (%d/%d)(%.4f). Best epoch: %d" % \
                                        (self.counts_not_improving, 
                                        self.max_counts_not_improving, 
                                        self.best_macro_fscore,
                                        self.best_epoch))
                if (self.with_early_stop and self.counts_not_improving >= self.max_counts_not_improving):
                    print ("Early stop")
                    f_val_target.close()
                    f_val_source.close()
                    return

        f_val_target.close()
        f_val_source.close()
        

        '''
        callbacks = []
        Fscore = Fscore()
        checkpoint = ModelCheckpoint(self.pathfile_saved_model, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='min', period=1)
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=2, mode='min')
        callbacks.append(checkpoint)
        callbacks.append(earlyStop)
        callbacks.append(Fscore)
        
        

        self.model.fit(
                    x=X_train_source,
                    y=Y_train_source,
                    validation_data=(X_train_target, Y_train_target),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1)
        '''

        
    def testDocuments(
                self,
                docs_test,
                model_config, 
                with_color, equalization_mode,
                mean_X_train, std_X_train,
                block_size,
                with_post_processing):
        assert(isinstance(model_config, ModelConfig))


        for doc_test in docs_test:
            src_im = FileManager.loadImage(doc_test.src_pathfile, with_color)
            src_im = apply_equalization(src_im, equalization_mode)
            gt_im = FileManager.loadImageFromPath(doc_test.gt_pathfile, False)
            print (doc_test.gt_pathfile)
            assert(np.min(gt_im) >= 0 and np.max(gt_im) <=1)
            gt_im = gt_im * 255

            norm_src_im = model_config.applyNormalizationDataset(imgs=src_im, mean = mean_X_train, std = std_X_train)
            norm_gt_im = (gt_im<128)
            norm_gt_im = to_categorical(norm_gt_im)

            x_sample_izq_sup = 0
            y_sample_izq_sup = 0

            [ROWS, COLS, DEPTH] = src_im.shape

            src_samples = []
            gt_samples = []

            prediction_full_src = np.zeros((ROWS, COLS))

            sample_height = block_size[0]
            sample_width = block_size[1]
            num_samples = len(range(0, ROWS-sample_height-1, sample_height)) * len(range(0, COLS - sample_width-1, sample_width))
            progress_bar = progressbar.ProgressBar(maxval=num_samples, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            print ("Loading samples. (%d samples)" % (num_samples))
            progress_bar.start()

            idx_sample = 0
            for x_sample_izq_sup in range(0, ROWS-1, sample_height):
                for y_sample_izq_sup in range(0, COLS-1, sample_width):

                    if (ROWS - 1 - x_sample_izq_sup<sample_height):
                        x_sample_izq_sup = ROWS - 1 - x_sample_izq_sup

                    if (COLS - 1 - y_sample_izq_sup<sample_width):
                        y_sample_izq_sup = COLS - 1 - y_sample_izq_sup

                    src_sample = norm_src_im[x_sample_izq_sup: x_sample_izq_sup + sample_height, y_sample_izq_sup:y_sample_izq_sup + sample_width]
                    gt_sample = norm_gt_im[x_sample_izq_sup: x_sample_izq_sup + sample_height, y_sample_izq_sup:y_sample_izq_sup + sample_width]

                    src_samples.append(src_samples)
                    gt_samples.append(gt_samples)
                    list_samples = [src_sample]
                    list_samples = np.asarray(list_samples).reshape(len(list_samples), block_size[0], block_size[1], block_size[2])

                    prediction_src_sample = self.model.predict(x=list_samples, batch_size=model_config.getBatchSize(), verbose=1)
                    binarized_prediction = np.argmax(prediction_src_sample, axis=3)
                    prediction_full_src[x_sample_izq_sup: x_sample_izq_sup + sample_height, y_sample_izq_sup:y_sample_izq_sup + sample_width] = binarized_prediction

                    idx_sample = idx_sample + 1
                    progress_bar.update(idx_sample)
                    
                
            progress_bar.finish()

            prediction_image = (1-prediction_full_src) * 255
            assert(np.min(prediction_full_src)>=0 and np.max(prediction_full_src) <=1)
            filename = FileManager.nameOfFileWithExtension(doc_test.gt_pathfile)
            FileManager.saveImageFullPath(prediction_image, self.pathdir_testing_images + "/"+filename)

            val_bbox = 127
            bbox_prediction_image = getBoundingBoxes(prediction_image, val=val_bbox)
            drawing = np.zeros((prediction_image.shape[0], prediction_image.shape[1], 3), dtype=np.uint8)
    
            for i in range(len(bbox_prediction_image)):
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                cv.rectangle(drawing, (int(bbox_prediction_image[i][0]), int(bbox_prediction_image[i][1])), \
                        (int(bbox_prediction_image[i][0]+bbox_prediction_image[i][2]), int(bbox_prediction_image[i][1]+bbox_prediction_image[i][3])), color, 2)

            FileManager.saveImageFullPath(drawing, self.pathdir_testing_images + "/bbox/" + filename)

            if with_post_processing:
                post_proccesed_prediction_image = prediction_image
                post_proccesed_prediction_image = cv2.morphologyEx(post_proccesed_prediction_image, cv2.MORPH_CLOSE, self.kernel_shape)
                iterations = 25
                post_proccesed_prediction_image = cv2.erode (post_proccesed_prediction_image, self.kernel_shape, iterations = iterations)
                post_proccesed_prediction_image = cv2.dilate(post_proccesed_prediction_image, self.kernel_shape, iterations = iterations)

                post_proccesed_prediction_image = cv2.morphologyEx(post_proccesed_prediction_image, cv2.MORPH_CLOSE, self.kernel_shape)
                #post_proccesed_prediction_image = cv2.morphologyEx(post_proccesed_prediction_image, cv2.MORPH_RECT, self.kernel_shape)

                filename_postprocessed = filename + "_post.png"
                FileManager.saveImageFullPath(post_proccesed_prediction_image, self.pathdir_testing_images + "/"+filename_postprocessed)

                bbox_prediction_image = getBoundingBoxes(post_proccesed_prediction_image, val=val_bbox)
                drawing = np.zeros((post_proccesed_prediction_image.shape[0], post_proccesed_prediction_image.shape[1], 3), dtype=np.uint8)
    
                for i in range(len(bbox_prediction_image)):
                    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                    cv.rectangle(drawing, (int(bbox_prediction_image[i][0]), int(bbox_prediction_image[i][1])), \
                            (int(bbox_prediction_image[i][0]+bbox_prediction_image[i][2]), int(bbox_prediction_image[i][1]+bbox_prediction_image[i][3])), color, 2)

                FileManager.saveImageFullPath(drawing, self.pathdir_testing_images + "/bbox/" + filename_postprocessed)


            prediction_full_src_cat = to_categorical(prediction_full_src,num_classes=self.num_classes)
            
            [binarized_imgs_1D, labels_test_cat_1D] = prepare_imgs_to_sklearn(prediction_full_src_cat, norm_gt_im, axis=2)
            macro_fscore = macroFScore(binarized_imgs_1D, labels_test_cat_1D)

            macro_fscore_all_classes = FScoreAllClasses(binarized_imgs_1D, labels_test_cat_1D, self.num_classes)
            assert(len(macro_fscore_all_classes) == self.num_classes)
            print("Fscore in classes: " + str(macro_fscore_all_classes))

            print ("MacroF1: " + str(macro_fscore))


    def test(self, X_test, Y_test, model_config, mean_X_train_source=None, std_X_train_source=None):

        assert(isinstance(model_config, ModelConfig))
        X_test = model_config.applyNormalizationDataset(X_test, mean_X_train_source, std_X_train_source)
        Y_test = (Y_test < 128)
        Y_test = to_categorical(y=Y_test, num_classes=self.num_classes)
        
        prediction = self.model.predict(x=X_test, batch_size=model_config.getBatchSize(), verbose=1)
        binarized_imgs = np.argmax(prediction, axis=3)
        evaluation = self.model.evaluate(x=X_test, y=Y_test, batch_size=model_config.getBatchSize(), verbose=1)
        
        test_acc = np.mean(binarized_imgs == (np.argmax(Y_test, axis=3)))

        print ( "[SAE - acc: %5f]" % (test_acc))
        print("Test evaluation")

        s = ""
        for idx_metric in range(len(self.model.metrics_names)):
            metric_name = self.model.metrics_names[idx_metric]
            result = evaluation[idx_metric]
            s = s + "(%s - %5f)" %(metric_name, result)
        print(s)
            
        [binarized_imgs_1D, labels_target_cat_1D] = prepare_imgs_to_sklearn(prediction, Y_test)
        macro_fscore = macroFScore(binarized_imgs_1D, labels_target_cat_1D)
        macro_precision = macroPrecision(binarized_imgs_1D, labels_target_cat_1D)
        macro_recall = macroRecall(binarized_imgs_1D, labels_target_cat_1D)
        fscore_values = FScoreValues(fscore=macro_fscore, precision=macro_precision, recall=macro_recall)

        print ("Fscore: " + str(fscore_values))

        if (self.with_saving_images):
            self.sample_images(is_testing=True, imgs=X_test, labels= Y_test, binarized_imgs=binarized_imgs, model_config=model_config, epoch=None)


    def load_model(self):
        self.model = load_model(self.pathfile_saved_model)



    def sample_images(self, is_testing, imgs, labels, binarized_imgs, model_config, epoch = None):
        r, c = 6, 8

        binarized_img3D = np.zeros((16, self.img_rows, self.img_cols, self.channels))
        labels_target_img3D = np.zeros((16, self.img_rows, self.img_cols, self.channels))
        idx = 0

        denorm_imgs = model_config.applyDeNormalization(imgs[0:16])
        denorm_binarized_imgs = (1-binarized_imgs[0:16,:,:]) * 255
        denorm_labels = (1-labels[0:16,:,:]) * 255
        

        for channel in range(self.channels):
            binarized_img3D[:,:,:,channel] = denorm_binarized_imgs
            labels_target_img3D[:,:,:,channel] = denorm_labels.astype(np.uint8)
            idx = idx + 1

        gen_imgs = np.concatenate([denorm_imgs, labels_target_img3D, binarized_img3D])
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

        if is_testing:
            pathdir = self.pathdir_testing_images
        else:
            pathdir = self.pathdir_training_images

        FileManager.makeDirsIfNeeded(pathdir)

        if (epoch is None):
            fig.savefig("%s/SAE.png" % (pathdir))
        else:
            fig.savefig("%s/%d.png" % (pathdir, epoch))
        plt.close()





if __name__ == '__main__':
    
    pathdir_exec = os.path.dirname(os.path.abspath(__file__))
    pathdir_gt_files_source = pathdir_exec + "/" + "databases/datasets_bin/GT/Einsiedeln"
    pathdir_src_files_source = pathdir_exec + "/" + "databases/datasets_bin/SRC/Einsiedeln"

    pathdir_gt_files_target = pathdir_exec + "/" + "databases/datasets_bin/GT/Salzinnes"
    pathdir_src_files_target = pathdir_exec + "/" + "databases/datasets_bin/SRC/Salzinnes"

    import os
    from keras import backend as K

    gpu = 0

    if K.backend() == 'tensorflow':
        import tensorflow as tf    # Memory control with Tensorflow
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        session_tf = tf.compat.v1.Session(config=config)
        K.set_session(session_tf)
    
        if (gpu is not None):
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]="0"
            
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    num_samples = 10000
    with_color = True
    folds = 1

    channels = 3 if with_color else 1
    block_size = (256, 256, channels)

    [training_docs_folds_source, test_docs_folds_source, validation_docs_folds_source] = lst_pathfiles_folds(pathdir_src_files_source, pathdir_gt_files_source, folds, 0.2, 0.2)
    [training_docs_folds_target, test_docs_folds_target, validation_docs_folds_target] = lst_pathfiles_folds(pathdir_src_files_target, pathdir_gt_files_target, folds, 0.2, 0.2)

    for fold in range(folds):

        [X_train_source, Y_train_source] = getDataSet(training_docs_folds_source[fold], num_samples, with_color, block_size)
        [X_train_target, Y_train_target] = getDataSet(training_docs_folds_target[fold], num_samples, with_color, block_size)
        [X_target_val, Y_target_val] = getDataSet(validation_docs_folds_target[fold], num_samples, with_color, block_size)

        X_train_source = np.asarray(X_train_source).reshape(len(X_train_source), block_size[0], block_size[1], block_size[2])
        Y_train_source = np.asarray(Y_train_source).reshape(len(Y_train_source), block_size[0], block_size[1], 1)

        X_train_target = np.asarray(X_train_target).reshape(len(X_train_target), block_size[0], block_size[1], block_size[2]) 
        Y_train_target = np.asarray(Y_train_target).reshape(len(Y_train_target), block_size[0], block_size[1], 1) 
        
        X_target_val = np.asarray(X_target_val).reshape(len(X_target_val), block_size[0], block_size[1], block_size[2]) 
        Y_target_val = np.asarray(Y_target_val).reshape(len(Y_target_val), block_size[0], block_size[1], 1) 

        sae = SAE(
                    input_shape=block_size,
                    kernel_shape=(3,3),
                    n_filters=128,
                    pool=2,
                    bn_axis=bn_axis,
                    activation_function='relu',
                    with_batch_normalization=False,
                    num_blocks=3,
                    pathfile_saved_model="saved_models/sae_model.h5")

        try:
            sae.model = load_model('saved_models/sae_model.h5')
        except:
            sae.train(
                        X_train_source = X_train_target, Y_train_source = Y_train_target,
                        X_train_target = X_target_val, Y_train_target = Y_target_val,
                        epochs=10, batch_size=32)

        sae.test(X_test=X_train_source, Y_test=Y_train_source, batch_size=32)