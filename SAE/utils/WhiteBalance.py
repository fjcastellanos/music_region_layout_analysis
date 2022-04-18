import numpy as np
import cv2

def whiteBalanceForRGBImage(image, perc = 0.05):
    return np.dstack([whiteBalanceForChannelImage(channel, 0.05) for channel in cv2.split(image)] )

def whiteBalanceForChannelImage(channel, perc = 0.05):

    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel


if __name__ == '__main__':

    from file_manager import FileManager

    im1 = FileManager.loadImage("databases/datasets_bin/SRC/Dibco/2010/aug_handwritten_GR/H02_v.png", True)
    im2 = FileManager.loadImage("databases/datasets_bin/SRC/Dibco/2010/aug_handwritten_GR/H04_o_zo.png", True)
    im3 = FileManager.loadImage("databases/datasets_bin/SRC/Salzinnes/train/sal_GR/CF-011.png", True)

    im1_wb = whiteBalanceForRGBImage(im1)
    FileManager.saveImage(im1, "prueba", "prueba1_before_wb.png")
    FileManager.saveImage(im1_wb, "prueba", "prueba1_after_wb.png")

    im2_wb = whiteBalanceForRGBImage(im2)
    FileManager.saveImage(im2, "prueba", "prueba2_before_wb.png")
    FileManager.saveImage(im2_wb, "prueba", "prueba2_after_wb.png")

    im3_wb = whiteBalanceForRGBImage(im3)
    FileManager.saveImage(im3, "prueba", "prueba3_before_wb.png")
    FileManager.saveImage(im3_wb, "prueba", "prueba3_after_wb.png")