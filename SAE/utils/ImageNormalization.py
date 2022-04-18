
import numpy as np


def normalization255(imgs):
    return imgs.astype(np.float32) / 255.

def denormalization255(imgs):
    denorm_imgs = imgs * 255
    return denorm_imgs.astype(np.uint8)

def normalizationInverse255(imgs):
    return (255. - imgs.astype(np.float32)) / 255.

def denormalizationInverse255(imgs):
    denorm_imgs = (1-imgs) * 255
    return denorm_imgs.astype(np.uint8)


def normalizationStandard(imgs):
    mean = np.mean(imgs)
    std = np.std(imgs)
    return normalizationStandardWithParams(imgs, mean, std)

def normalizationStandardWithParams(imgs, mean, std):
    return (imgs.astype(np.float32) - mean) / (std + 0.00001)

def denormalizationStandardWithParams(imgs, mean, std):
    denorm_imgs = imgs * (std + 0.00001) + mean
    return denorm_imgs.astype(np.uint8)

def normalizationMean(imgs):
    mean = np.mean(imgs)
    return normalizationMeanWithParams(imgs, mean)

def normalizationMeanWithParams(imgs, mean):
    return imgs.astype(np.float32) - mean

def denormalizationMeanWithParams(imgs, mean):
    denorm_imgs =  imgs + mean
    return denorm_imgs.astype(np.uint8)