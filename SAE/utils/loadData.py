
# ----------------------------------------------------------------------------
def load_dataset_folds(dbname, dbparam):
    train_folds = []
    test_folds = []
    
    DIBCO = [    ['Dibco/2009/handwritten_GR', 'Dibco/2009/printed_GR'],
                 ['Dibco/2010/handwritten_GR'],
                 ['Dibco/2011/handwritten_GR', 'Dibco/2011/printed_GR'],
                 ['Dibco/2012/handwritten_GR'],
                 ['Dibco/2013/handwritten_GR', 'Dibco/2013/printed_GR'],
                 ['Dibco/2014/handwritten_GR'],
                 ['Dibco/2016/handwritten_GR']     ]
    
    PALM_train = [ ['Palm/Challenge-1-ForTrain/gt1_GR'], ['Palm/Challenge-1-ForTrain/gt2_GR'] ]
    PALM_test = [ ['Palm/Challenge-1-ForTest/gt1_GR'], ['Palm/Challenge-1-ForTest/gt2_GR'] ]
    
    PHI_train = ['PHI/train/phi_GR']
    PHI_test = ['PHI/test/phi_GR']
    
    EINSIELDELN_train = ['Einsieldeln/train/ein_GR']
    EINSIELDELN_test = ['Einsieldeln/test/ein_GR']
    
    SALZINNES_train = ['Salzinnes/train/sal_GR']
    SALZINNES_test = ['Salzinnes/test/sal_GR']
    
    VOYNICH_test = ['Voynich/voy_GR']
    
    BDI_train = ['BDI/train/bdi11_GR']
    BDI_test = ['BDI/test/bdi11_GR']

    if dbname == 'custom':
        train_folds = None
        test_folds = ['Custom']
    elif dbname == 'dibco':
        dbparam = int(dbparam)
        test_folds = DIBCO[dbparam]
        DIBCO.pop(dbparam)
        train_folds = [val for sublist in DIBCO for val in sublist]
    elif dbname == 'palm':
        dbparam = int(dbparam)
        train_folds = PALM_train[dbparam]
        test_folds = PALM_test[dbparam]
    elif dbname == 'phi':
        train_folds = PHI_train
        test_folds = PHI_test
    elif dbname == 'ein':
        train_folds = EINSIELDELN_train
        test_folds = EINSIELDELN_test
    elif dbname == 'sal':
        train_folds = SALZINNES_train
        test_folds = SALZINNES_test
    elif dbname == 'voy':
        train_folds = [val for sublist in DIBCO for val in sublist]
        test_folds = VOYNICH_test
    elif dbname == 'bdi':
        train_folds = BDI_train
        test_folds = BDI_test
    elif dbname == 'all':
        test_folds = [DIBCO[5], DIBCO[6]]
        test_folds.append(PALM_test[0])
        test_folds.append(PALM_test[1])
        test_folds.append(PHI_test)
        test_folds.append(EINSIELDELN_test)
        test_folds.append(SALZINNES_test)
        
        DIBCO.pop(6)
        DIBCO.pop(5)        
        train_folds = [[val for sublist in DIBCO for val in sublist]]
        train_folds.append(PALM_train[0])
        train_folds.append(PALM_train[1])
        train_folds.append(PHI_train)
        train_folds.append(EINSIELDELN_train)
        train_folds.append(SALZINNES_train)
        
        test_folds = [val for sublist in test_folds for val in sublist]  # transform to flat lists
        train_folds = [val for sublist in train_folds for val in sublist]
    else:
        raise Exception('Unknown database name')
    
    return train_folds, test_folds



# ----------------------------------------------------------------------------
def load_dataset_folds_GT(dbname, dbparam):
    
    train_folds, test_folds = load_dataset_folds(dbname, dbparam)
    train_folds = [train_folds.replace('GR', 'GT') for train_folds in train_folds]
    test_folds = [test_folds.replace('GR', 'GT') for test_folds in test_folds]

    return train_folds, test_folds

