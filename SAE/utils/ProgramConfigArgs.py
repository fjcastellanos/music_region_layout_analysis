import cv2
import os, sys
import numpy as np
from enum import Enum


#==============================================================================
#   Sub arguments
#==============================================================================
def addCoordsRect(parser):
    parser.add_argument('--xini', help='Initial position x to test', dest = 'x_ini', type=int, required=True)
    parser.add_argument('--xend', help='End position x to test', dest = 'x_end', type=int, required=True)    
    parser.add_argument('--yini', help='Initial position y to test', dest = 'y_ini', type=int, required=True)
    parser.add_argument('--yend', help='End position y to test', dest = 'y_end', type=int, required=True)
        
    
def addSpan(parser):
    parser.add_argument('--vspan', help='vertical span per block', dest = 'v_span', type=int, required=True)
    parser.add_argument('--hspan', help='horizontal span per block', dest = 'h_span', type=int, required=True)
    
    
def addCNNOptions(parser):
    parser.add_argument('--speed', help='speed factor to train faster', dest = 'factor_speed', type=int, default=1)
    parser.add_argument('--ncodes', help='number of neural codes', dest = 'n_codes', type=int, default=64)
    parser.add_argument('--cnn', dest='cnn', help='CNN model', choices=['advanced', 'medium', 'medium2', 'small', 'myresnet2', 'resnet50', 'sae'], required=True)
    parser.add_argument('--opt', const='adadelta', nargs='?', help='Options for training thes model', choices=['adadelta', 'sgd'], default='adadelta')
    
def addCallbackType(parser):
    parser.add_argument('--callback', dest='callback', help='CNN callback', choices=['normal', 'F1validation', 'mse'], required=True)
    
    
    
    
def addDBName(parser):
    parser.add_argument('--db', dest='db', help='Name of database', choices=['einsiedeln', 'salzinnes'], required=True)
   
    

def addGPUIdx(parser):
    parser.add_argument('--gpu', help='Identifier of GPU', dest='gpu', type=int, required=True)
        


class ArgumentType(Enum):
    TRAINING, \
    TEST, \
    EXCHANGELABELS, \
    COLORING, \
    CV, \
    STUDY, \
    STUDYDB, \
    LOADJSONCNNS, \
    CHECKMODELS, \
    SCALE_FIXER, \
    ENTROPY_HISTOGRAM,\
    FMEASURE,\
    PARSER,\
    PREDICT_COMPARE,\
    PREDICT_COMBINE,\
    GAN\
    = range(16)


def argumentToString(argument_type):
    assert argument_type.value < len(ArgumentType)
    
    return {
                ArgumentType.TRAINING:           "TRAINING",
                ArgumentType.TEST:               "TEST",
                ArgumentType.EXCHANGELABELS:     "EXCHANGELABELS",
                ArgumentType.COLORING:           "COLORING",
                ArgumentType.CV:                 "CV",
                ArgumentType.STUDY:              "STUDY",
                ArgumentType.STUDYDB:            "STUDYDB",
                ArgumentType.CHECKMODELS:        "CHECKMODELS",
                ArgumentType.SCALE_FIXER:        "SCALE_FIXER",
                ArgumentType.FMEASURE:           "FMEASURE",
                ArgumentType.PARSER:             "PARSER",
                ArgumentType.PREDICT_COMPARE:    "PREDICT_COMPARE",
                ArgumentType.PREDICT_COMBINE:    "PREDICT_COMBINE",
                ArgumentType.GAN:            "GAN"
                }[argument_type]
                

def argumentToCommand(argument_type):
    assert argument_type.value < len(ArgumentType)
    
    return {
                ArgumentType.TRAINING:           "training",
                ArgumentType.TEST:               "test",
                ArgumentType.EXCHANGELABELS:     "exchangelabels",
                ArgumentType.COLORING:           "coloring",
                ArgumentType.CV:                 "cv",
                ArgumentType.STUDY:              "study",
                ArgumentType.STUDYDB:            "studydb",
                ArgumentType.CHECKMODELS:        "check",
                ArgumentType.SCALE_FIXER:        "scalefixer",
                ArgumentType.FMEASURE:           "fmeasure",
                ArgumentType.PARSER:             "parser",
                ArgumentType.PREDICT_COMPARE:    "predict_compare",
                ArgumentType.PREDICT_COMBINE:    "predict_combine",
                ArgumentType.GAN:                "gan"
                }[argument_type]
  
    
def argumentCompareCommand(argument_type, input_command):
    assert argument_type.value < len(ArgumentType)
    
    if (argumentToCommand(argument_type) == input_command) :
        return True
    else:
        return False
    
    

class ProgramConfigArgs:
    
    ptrs_args = {}
    
    
#==============================================================================
#     Arguments
#==============================================================================
    

    def addArgumentFileConfig(self, parsers):
        parser = parsers.add_parser('fileconfig')
        parser.add_argument('--f', help='File with configuration about the executable', dest='file_config')
    
        
        
    def addArgumentStudy(self, parsers):
        parser = parsers.add_parser('study')
        addDBName(parser)
        parser.add_argument('--model', help='Path to model file', dest='model')
        parser.add_argument('--samples', help='number of samples to study', dest='samples', type=int, required=True)
        parser.add_argument('--fold', help='Index of fold >= 0', dest='fold', type=int, required=True)
        addSpan(parser)
        parser.add_argument('--speed', help='speed factor to get process faster', dest = 'speed', type=int, default=1)
        addGPUIdx(parser)        
    
    def addArgumentStudyDB(self, parsers):
        parser = parsers.add_parser('studydb')
        addDBName(parser)
        parser.add_argument('--samples', help='number of samples to study', dest='samples', type=int, required=True)
        parser.add_argument('--fold', help='Index of fold >= 0', dest='fold', type=int, required=True)
        parser.add_argument('--speed', help='speed factor to get process faster', dest = 'speed', type=int, default=1)
        
        
    def addArgumentTraining(self, parsers):
        parser = parsers.add_parser('training')
        addSpan(parser)
        addCNNOptions(parser)
        parser.add_argument('--samples', help='number of samples per class', dest='samples_per_class', type=int, required=True)
        addCoordsRect(parser)
        addGPUIdx(parser)
    
    def addArgumentTest(self, parsers):
        parser = parsers.add_parser('test')
        addSpan(parser)
        addCoordsRect(parser)
        addCNNOptions(parser)
        addGPUIdx(parser)
    
    def addArgumentExchangeLabels(self, parsers):
        parser = parsers.add_parser('exchangelabels')
        addDBName(parser)
        
        
    def addArgumentColoringGT(self, parsers):
        parser = parsers.add_parser('coloring')
        addCoordsRect(parser)
        
        parser.add_argument('--f', help='Labeled GT file name for coloring', dest='file')
        
                    
    def addArgumentCrossValidation(self, parsers):
        parser = parsers.add_parser('cv')
        parser.add_argument('--folds', help='number of folds per class', dest='folds', type=int, required=True)
        addSpan(parser)
        addCNNOptions(parser)
        #addCoordsRect(parser)
        parser.add_argument('--samples', help='number of samples per class', dest='samples_per_class', type=int, required=True)
        parser.add_argument('--no_training', help='Disable training module', dest='no_training', required=False, action='store_true')
        addCallbackType(parser)
        addGPUIdx(parser)
        
        
    def addArgumentsLoadJSONCNNs(self, parsers):
        parser = parsers.add_parser('jsoncnn')
        parser.add_argument('--folds', help='number of folds per class', dest='folds', type=int, required=True)
        addCallbackType(parser)
        parser.add_argument('--samples_test', help='number of random samples per test', dest='samples_test', type=int, required=True)
        parser.add_argument('--f', help='File with configuration about the executable', dest='file_config')
        parser.add_argument('--mincnn', help='Minimum CNN index to evaluate', dest='mincnn', type=int, required=False)
        parser.add_argument('--maxcnn', help='Maximum CNN index to evaluate', dest='maxcnn', type=int, required=False)
        parser.add_argument('--no_training', help='Disable training module', dest='no_training', required=False, action='store_true')
        parser.add_argument('--no_test', help='Disable test module', dest='no_test', required=False, action='store_true')
        parser.add_argument('--no_repeat', help='Disable repetition of training models', dest='no_repeat', required=False, action='store_true')
        parser.add_argument('--retrain', help='Retrain a trained model', dest='retrain', required=False, action='store_true')
        addDBName(parser)
        addGPUIdx(parser)
        
        
    def addArgumentsCheckModels(self, parsers):
        parser = parsers.add_parser('check')
        addCallbackType(parser)
        parser.add_argument('--f', help='File with configuration about the executable', dest='file_config')
        parser.add_argument('--mincnn', help='Minimum CNN index to evaluate', dest='mincnn', type=int, required=False)
        parser.add_argument('--maxcnn', help='Maximum CNN index to evaluate', dest='maxcnn', type=int, required=False)
        addDBName(parser)
    
    def addArgumentsScaleFixer(self, parsers):
        parser = parsers.add_parser('scalefixer')
        parser.add_argument('--f-as', help='File with configuration about the executable (AUTOSCALER)', dest='file_config_as')
        parser.add_argument('--f-sae', help='File with configuration about the executable (SAE)', dest='file_config_sae')
        parser.add_argument('--mincnn-as', help='Minimum CNN index to evaluate', dest='mincnn_as', type=int, required=False)
        parser.add_argument('--maxcnn-as', help='Maximum CNN index to evaluate', dest='maxcnn_as', type=int, required=False)
        parser.add_argument('--mincnn-sae', help='Minimum CNN index to evaluate', dest='mincnn_sae', type=int, required=False)
        parser.add_argument('--maxcnn-sae', help='Maximum CNN index to evaluate', dest='maxcnn_sae', type=int, required=False)
        
        parser.add_argument('--no_training', help='Disable training module', dest='no_training', required=False, action='store_true')
        parser.add_argument('--no_test', help='Disable test module', dest='no_test', required=False, action='store_true')
        parser.add_argument('--with_sae', help='Enable SAE module', dest='with_sae', required=False, action='store_true')
        parser.add_argument('--train_sae', help='Enable SAE training', dest='train_sae', required=False, action='store_true')
        
        parser.add_argument('--samples_test', help='number of random samples per test', dest='samples_test', type=int, required=True)
        parser.add_argument('--folds', help='number of folds per class', dest='folds', type=int, required=True)
        parser.add_argument('--start_fold', help='Start index fold', dest='start_fold', type=int, required=False)
        parser.add_argument('--end_fold', help='End index fold', dest='end_fold', type=int, required=False)
        parser.add_argument('--single_scale', help='Scale for considered documents in testing step', dest='single_scale', type=float, required=False)
        
        addDBName(parser)
        addGPUIdx(parser)
        
        parser.add_argument('--seed', help='Seed for random function', dest='seed', type=int, required=False)
    
    def addArgumentsEntropyHistogram(self, parsers):
        parser = parsers.add_parser('entropy-hs')
        
        addDBName(parser)
        parser.add_argument('--height_block', help='Block height to calculate entropy', dest='height_block', type=int, required=True)
        parser.add_argument('--width_block', help='Block width to calculate entropy', dest='width_block', type=int, required=True)
        parser.add_argument('--num_ranges', help='Number of ranges for histogram', dest='num_ranges', type=int, required=True)
        parser.add_argument('--max_entropy', help='Maximum value for entropy in the histogram', dest='max_entropy', type=float, required=True)
        parser.add_argument('--v', help='Activate verbose', dest='v', required=False, action='store_true')
        addGPUIdx(parser)
        
    def addArgumentFMeasure(self, parsers):
        parser = parsers.add_parser('fmeasure')
        
        parser.add_argument('--f-pred', help='File with prediction', dest='file_pred', required=True)
        parser.add_argument('--f-gt', help='File with labeled image', dest='file_gt', required=True)
        parser.add_argument('--n_classes', help='Number of classes', dest='n_classes', type=int, required=True)
        
    def appendToParser(self, parser, name_argument):
        assert name_argument.value < len(ArgumentType)
        self.ptrs_args[name_argument](parser)
        
    
    def addArgumentParser(self, parsers):
        parser = parsers.add_parser('parser')
        parser.add_argument('--f', help='File with log results', dest='file')
    
    def addArgumentPredictCompare(self, parsers):
        parser = parsers.add_parser('predict_compare')
        parser.add_argument('--f-pred', help='Prediction GT file', dest='f_pred')
        parser.add_argument('--f-exp', help='Expected GT file', dest='f_exp')
    
    def addArgumentPredictCombine(self, parsers):
        parser = parsers.add_parser('predict_combine')
        parser.add_argument('--f-0', help='Prediction GT file with 0 class', dest='f_0', required=True)
        parser.add_argument('--f-1', help='Prediction GT file with 1 class', dest='f_1', required=True)
        parser.add_argument('--f-2', help='Prediction GT file with 2 class', dest='f_2', required=False)
        parser.add_argument('--f-3', help='Prediction GT file with 3 class', dest='f_3', required=False)
        
        parser.add_argument('--f-out', help='Combination file', dest='f_out', required=True)
        
    def addArgument_GAN(self, parsers):
        parser = parsers.add_parser('gan')
        
        parser.add_argument('--f-gen', help='File with configuration about the generator model', dest='file_config_gen')
        parser.add_argument('--f-dis', help='File with configuration about the discriminator model', dest='file_config_dis')
        parser.add_argument('--mincnn-gen', help='Minimum CNN index to evaluate from generator models', dest='mincnn_gen', type=int, required=False)
        parser.add_argument('--maxcnn-gen', help='Maximum CNN index to evaluate from generator models', dest='maxcnn_gen', type=int, required=False)
        parser.add_argument('--mincnn-dis', help='Minimum CNN index to evaluate from discriminator models', dest='mincnn_dis', type=int, required=False)
        parser.add_argument('--maxcnn-dis', help='Maximum CNN index to evaluate from discriminator models', dest='maxcnn_dis', type=int, required=False)
        
        parser.add_argument('--exe_mode', dest='exe_mode', help='Execution mode', choices=['train', 'test'], required=True)

        parser.add_argument('--train_dbname', help='Name of the training dataset', dest='train_dbname', required=True)
        parser.add_argument('--test_dbname', help='Name of the testing dataset', dest='test_dbname', required=True)

        
        parser.add_argument('--total_samples', help='number of random samples per test', dest='total_samples', type=int, required=True)
        parser.add_argument('--start_fold', help='Start index fold', dest='start_fold', type=int, required=False)
        parser.add_argument('--end_fold', help='End index fold', dest='end_fold', type=int, required=False)
        
        addDBName(parser)
        addGPUIdx(parser)
        


    ptrs_args = {
                ArgumentType.TRAINING:           addArgumentTraining,
                ArgumentType.TEST:               addArgumentTest,
                ArgumentType.EXCHANGELABELS:     addArgumentExchangeLabels,
                ArgumentType.COLORING:           addArgumentColoringGT,
                ArgumentType.CV:                 addArgumentCrossValidation,
                ArgumentType.STUDY:              addArgumentStudy,
                ArgumentType.STUDYDB:            addArgumentStudyDB,
                ArgumentType.LOADJSONCNNS:       addArgumentsLoadJSONCNNs,
                ArgumentType.CHECKMODELS:        addArgumentsCheckModels,
                ArgumentType.SCALE_FIXER:        addArgumentsScaleFixer,
                ArgumentType.ENTROPY_HISTOGRAM:  addArgumentsEntropyHistogram,
                ArgumentType.FMEASURE:           addArgumentFMeasure,
                ArgumentType.PARSER:             addArgumentParser,
                ArgumentType.PREDICT_COMPARE:    addArgumentPredictCompare,
                ArgumentType.GAN:                addArgument_GAN
                } 
                
    