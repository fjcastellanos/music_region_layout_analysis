import cv2
import numpy as np
import os

from sys import stdout

try:
    from image_manager import *
except:
    from utils.image_manager import *

try:
    from file_manager import FileManager
except:
    from utils.file_manager import FileManager



source_folder = 'Set'
files = ['add_ms_28966_f003r.png', 'add_ms_28966_f003v.png']
source_label = '00_original'

considered_labels = [
        '00_background',
        '01_text',
        '02_new_ink', 
        '03_old_ink', 
        '04_scribble'
        ]
        
original_labels = {
        '01_text':              1, 
        '02_scribble':          4, 
        '03_clef_key_draft':    3, 
        '04_staffline_draft':   0, 
        '05_glyph_draft':       3, 
        '06_barlines_draft':    0, 
        '07_clef_key_final':    3, 
        '08_staffline_final':   0, 
        '09_barlines_final':    0, 
        '10_glyphs_final':      2}
          

colour_dict = {}
colour_dict[0] = [0, 0, 0]
colour_dict[1] = [255, 255, 255]
colour_dict[2] = [255, 0, 0]
colour_dict[3] = [0, 255, 0]
colour_dict[4] = [0, 0, 255]
colour_dict[5] = [255, 255, 0]
colour_dict[6] = [255, 0, 255]
colour_dict[7] = [0, 255, 255]
colour_dict[8] = [128, 128, 128]
colour_dict[9] = [190, 190, 190]
colour_dict[10] = [128, 128, 0]
colour_dict[11] = [128, 0, 128]
colour_dict[12] = [0, 128, 128]



#==============================================================================
# def getLabeledImage(filename, source_label):
#     source_image = cv2.imread(source_folder + '/' + source_label + '/' + filename, True)
# 
#     gt_image = np.zeros( (source_image.shape[0], source_image.shape[1]), 'uint8')
#     
#     for original_label, idx_label in original_labels.items():
#         label_image = cv2.imread(source_folder + '/' + original_label + '/' + filename, False)
#  
#         print ('\t Processing ' + str(considered_labels[idx_label]))
#         
#         gt_image += (label_image < 230).astype('uint8') * (idx_label)
#         #cv2.imwrite(filename+'_'+str(label_idx)+'.png',(label_image < 230)*255.)
#         
#     return gt_image
#==============================================================================
    

def generateGT():
    for filename in files:
        print ('Processing ' + str(filename))
        source_image = FileManager.loadImage(source_folder + '/' + source_label + '/', filename, False)
        
        getLabelImage(filename, source_folder, source_image.shape[0], source_image.shape[1], original_labels, considered_labels)
        
        labeled_image = getLabeledImage(filename,source_label)
        labeled_image.dump(filename+'.dat')
        
def getColouredImageFromGT(gt_filename, x_ini, y_ini, x_end, y_end): 
    
    gt_image = FileManager.loadImageFromPath(gt_filename, False)
    
    coloured_image = np.zeros( (gt_image.shape[0], gt_image.shape[1], 3), 'uint8')
    
    for row in range(x_ini, x_end):
        for col in range(y_ini, y_end):
            size = gt_image.shape
            if (len(size) == 3):
                label = gt_image[row][col][0]
            else:
                label = gt_image[row][col]
                
            if not label in colour_dict:
                colour_dict[label] = (np.random.rand(3)*255).astype('uint8')
                
            coloured_image[row][col] = colour_dict[label]
    
    return coloured_image

def getColouredImageFromGT2(gt_image,x_ini, y_ini, x_end, y_end): 
    
    coloured_image = np.zeros( (gt_image.shape[0], gt_image.shape[1], 3), 'uint8')
    
    for row in range(x_ini, x_end):
        for col in range(y_ini, y_end):
            size = gt_image.shape
            if (len(size) == 3):
                label = gt_image[row][col][0]
            else:
                label = gt_image[row][col]
                
            if not label in colour_dict:
                colour_dict[label] = (np.random.rand(3)*255).astype('uint8')
                
            coloured_image[row][col] = colour_dict[label]
    
    return coloured_image
    





class ImageProcessing:
    
#==============================================================================
#     
#==============================================================================
    def getLabelImage(filename, source_parent_folder, width, height, original_labels, considered_labels):
        assert type(source_parent_folder) == str
        assert type(width) == int
        assert type(height) == int

        gt_image = np.zeros( (width, height), 'uint8')
        
        for original_label, idx_label in original_labels.items():
            label_image = FileManager.loadImage (source_parent_folder + '/' + original_label + '/', filename, False)
     
            print ('\t Processing ' + str(considered_labels[idx_label]))
            
            gt_image += (label_image < 230).astype('uint8') * (idx_label)
            #cv2.imwrite(filename+'_'+str(label_idx)+'.png',(label_image < 230)*255.)
            
        return gt_image


    getLabelImage = staticmethod(getLabelImage)


#==============================================================================
# 
#==============================================================================
    def countLabeledPixels(gt_image, h_span, v_span, considered_labels, factor_speed):
        print ('Counting pixels by label')
        #rescaled_image = gt_image[v_span:gt_image.shape[0]-v_span+1:i_FACTOR_SPEED, h_span:gt_image.shape[1]-h_span+1:i_FACTOR_SPEED]
        #print "Rescaled image: " + str(rescaled_image.shape) 
        #f = np.histogram(rescaled_image, bins=len(considered_labels))[0].tolist()    
        #print "Histogram: " + str(f)        
        #return f
        
        img = gt_image[v_span:gt_image.shape[0]-v_span+1,h_span:gt_image.shape[1]-h_span+1]
        
        
        counts = [0] * len(considered_labels)
        
        if (factor_speed == 1):
        
            for idx_label in range (len(considered_labels)):
                counts[idx_label] = sum(sum(img == idx_label))  
        else:
        
            for row in range(v_span, gt_image.shape[0]-v_span+1, factor_speed):
                for col in range(h_span, gt_image.shape[1]-h_span+1, factor_speed):
                    idx_label = gt_image[row][col]
                    if idx_label in range(len(considered_labels)):
                       counts[idx_label] = counts[idx_label] + 1
    
        print ("Manual histogram: " + str(counts[0:len(considered_labels)]))
        return counts[0:len(considered_labels)]
    

    countLabeledPixels = staticmethod(countLabeledPixels)


# =============================================================================
# 
# =============================================================================

    def countLabeledPixelsWithConfig(gt_image, corpus_config):
        print ('Counting pixels by label')
        #rescaled_image = gt_image[v_span:gt_image.shape[0]-v_span+1:i_FACTOR_SPEED, h_span:gt_image.shape[1]-h_span+1:i_FACTOR_SPEED]
        #print "Rescaled image: " + str(rescaled_image.shape) 
        #f = np.histogram(rescaled_image, bins=len(considered_labels))[0].tolist()    
        #print "Histogram: " + str(f)        
        #return f
        from corpus_config import CorpusConfig
        assert(isinstance(corpus_config, CorpusConfig))
        
        [height, width] = corpus_config.getDimensions()
        
        counts = [0] * len(corpus_config.considered_labels)
        
        if (corpus_config.factor_speed == 1):
            for idx_label in range (len(corpus_config.considered_labels)):
                counts[idx_label] = sum(sum(gt_image == idx_label))  
        else:
        
            for row in range(0, gt_image.shape[0] - height, corpus_config.factor_speed):
                for col in range(0, gt_image.shape[1] - width, corpus_config.factor_speed):
                    idx_label = gt_image[row][col]
                    if idx_label in range(len(corpus_config.considered_labels)):
                       counts[idx_label] = counts[idx_label] + 1
    
        print ("Manual histogram: " + str(counts[0:len(corpus_config.considered_labels)]))
        return counts[0:len(corpus_config.considered_labels)]
    

    countLabeledPixelsWithConfig = staticmethod(countLabeledPixelsWithConfig)


    def getProbabilitiesLabeledPixelsAllFiles(folder, files, max_blocks_per_class, h_span, v_span, original_labels, considered_labels):
    
        number_instances_per_class = [0] * len(considered_labels)

        for f in files:
            gt_filename = folder + "/" + f + '.dat'
            gt_image = np.load(gt_filename)
            counts = countLabeledPixels(gt_image, h_span, v_span, original_labels, considered_labels)
            number_instances_per_class = map(operator.add, number_instances_per_class, counts)
            
        print ('Total number of instances per class: ' + str(number_instances_per_class))
        min_count = min(number_instances_per_class)
        
        if max_blocks_per_class < min_count:
            min_count = max_blocks_per_class;
            
        probs = [0] * len(number_instances_per_class)
    
        for idx in range (len(number_instances_per_class)):
            probs[idx] = min_count / float(number_instances_per_class[idx]);
        
        print ('Probabilities: ' + str(probs))
        return probs
        
        
    getProbabilitiesLabeledPixelsAllFiles = staticmethod(getProbabilitiesLabeledPixelsAllFiles)
    



#==============================================================================
#     
#==============================================================================
    def exchangeLabelsFromOriginalToConsidered(path_dir, filename, original_labels):
        assert type(path_dir) == str
        assert type(filename) == str
        
        image = FileManager.loadImage(path_dir, filename, False)
        
        gt_image = np.zeros( (image.shape[0], image.shape[1]), 'uint8')
        
        
        
        orig_labels_ordered = sorted(original_labels)
        print ("ORIGINAL: " + str(orig_labels_ordered))
        print ("Starting translate to considered labels")
        for row in range(gt_image.shape[0]):
            for col in range(gt_image.shape[1]):
                idx_label = image[row][col]
                #if (idx_label >= 4):
                #    print str(idx_label) + "->" + str(original_labels[orig_labels_ordered[idx_label]])
                gt_image[row][col] = original_labels[orig_labels_ordered[idx_label]]
                #print("Orig:" + str(idx_label))
                #print("Consid:" + str(orig_labels_ordered[idx_label][1]))
        
        return gt_image
        
        print ("Ending translate to considered labels")


    
    exchangeLabelsFromOriginalToConsidered = staticmethod(exchangeLabelsFromOriginalToConsidered)
    
    
    
    
# =============================================================================
#     
# =============================================================================
    def scaleImageGTandSRC(image, gt_image, scale):
        if scale is None or scale == 1.:
            return [image.copy(), gt_image.copy()]
        else:
            scaled_image = scaleImage(image, scale)
            scaled_gt_image = scaleImage(gt_image, scale, cv2.INTER_NEAREST)
    
            return [scaled_image, scaled_gt_image]
    
    scaleImageGTandSRC = staticmethod(scaleImageGTandSRC)
    

    def redimAnyImage(image, height, width):
        return redimImage(image, height, width, cv2.INTER_NEAREST)
    
    redimAnyImage = staticmethod(redimAnyImage)

    
    
    
# =============================================================================
#     
# =============================================================================
    def entropyHistogram(image, height_block, width_block, num_ranges, max_entropy, verbose = False):
        
        precision = float(max_entropy) / float(num_ranges)
        hist = [0] * num_ranges
        num_pixels = image.shape[0] * image.shape[1]
        count = 0
        for row in range(image.shape[0] - height_block):
            for col in range(image.shape[1] - width_block):
                
                block = image[row: row + height_block - 1, col: col + width_block - 1]
                entropy = ImageProcessing.calculateEntropy(block)
                
                index = int(entropy / precision)
                hist[index] = hist[index] + 1
                
                if verbose == True:
                    count = count + 1
                    count_tpc = count * 100. / num_pixels
                    
                    stdout.write("\r%d of %d pixels (%d%%)" % (count, num_pixels, count_tpc))
                    stdout.flush()
            
        return hist
    
    entropyHistogram = staticmethod(entropyHistogram)
    
    
# =============================================================================
#     
# =============================================================================
    def calculateEntropy(image): 
        hist = cv2.calcHist([image],[0],None,[256],[0,256])
        hist = hist.ravel()/hist.sum()
        logs = np.log2(hist+0.00001)
        entropy = -1 * (hist*logs).sum()
        return entropy  

    calculateEntropy = staticmethod(calculateEntropy)






    
if __name__ == "__main__":
    print ('Generating')
    generateGT()

    for filename in files:
        print ('Colouring ' + str(filename))
        coloured_image = getColouredImageFromGT(filename+'.dat')
        cv2.imwrite('GT_'+filename,coloured_image)
