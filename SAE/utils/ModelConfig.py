from utils.ImageProcessing import *
from utils.ImageNormalization import *

TYPE_NORMALIZATION_255 = "255"
TYPE_NORMALIZATION_INVERSE_255 = "inv255"
TYPE_NORMALIZATION_STANDARD = "standard"
TYPE_NORMALIZATION_MEAN = "mean"
TYPE_NORMALIZATION_ORIGINAL = "original"

NORMALIZATION_LIST = [
            TYPE_NORMALIZATION_255, 
            TYPE_NORMALIZATION_INVERSE_255,
            TYPE_NORMALIZATION_STANDARD,
            TYPE_NORMALIZATION_MEAN, 
            TYPE_NORMALIZATION_ORIGINAL]

class ModelConfig:

    def _i_integrity(self):
        assert(type(self.type_normalization) is str)
        assert(type(self.epochs) is int)
        assert(type(self.batch_size) is int)

        assert(self.type_normalization in NORMALIZATION_LIST)
        assert(self.epochs  > 0)
        assert(self.batch_size  > 0)
        assert(self.considered_classes is None or type(self.considered_classes) is list)

    def __init__(self, type_normalization, epochs, batch_size=32, considered_classes=None):
        self.type_normalization = type_normalization
        self.epochs = epochs
        self.batch_size = batch_size
        self.considered_classes = considered_classes
        self._i_integrity()


    def applyNormalizationDataset(self, imgs, mean = None, std = None):
        self._i_integrity()

        if self.type_normalization == TYPE_NORMALIZATION_255:
            return normalization255(imgs)
        if (self.type_normalization == TYPE_NORMALIZATION_INVERSE_255):
            return normalizationInverse255(imgs)
        elif self.type_normalization == TYPE_NORMALIZATION_STANDARD:
            assert(mean is not None)
            assert(std is not None)
            return normalizationStandardWithParams(imgs, mean, std)
        elif self.type_normalization == TYPE_NORMALIZATION_MEAN:
            assert(mean is not None)
            return normalizationMeanWithParams(imgs, mean)
        elif self.type_normalization == TYPE_NORMALIZATION_ORIGINAL:
            return imgs
        else:
            assert(False)

        
    def applyDeNormalization(self, imgs, mean=None, std=None):
        self._i_integrity()

        if self.type_normalization == TYPE_NORMALIZATION_255:
            return denormalization255(imgs)
        if (self.type_normalization == TYPE_NORMALIZATION_INVERSE_255):
            return denormalizationInverse255(imgs)
        elif self.type_normalization == TYPE_NORMALIZATION_STANDARD:
            assert(mean is not None)
            assert(std is not None)
            return denormalizationStandardWithParams(imgs, mean, std)
        elif self.type_normalization == TYPE_NORMALIZATION_MEAN:
            assert(mean is not None)
            return denormalizationMeanWithParams(imgs, mean)
        elif self.type_normalization == TYPE_NORMALIZATION_ORIGINAL:
            return imgs
        else:
            assert(False)        


    def getNumberEpochs(self):
        self._i_integrity()
        return self.epochs

    def getBatchSize(self):
        self._i_integrity()
        return self.batch_size

    def getConsideredClasses(self):
        self._i_integrity()
        return self.considered_classes


        