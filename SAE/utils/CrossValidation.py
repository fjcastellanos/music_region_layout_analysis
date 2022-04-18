

class CrossValidation:

    training_documents = []
    test_documents = []
    validation_documents = []

    def _i_integrity(self):
        assert(type(self.training_files_folds) is list)
        assert(type(self.test_files_folds) is list)
        assert(type(self.training_files_folds) is list)
        assert(len(self.validation_files_folds) == len(self.test_files_folds) == len(self.validation_files_folds) == self.folds)

    def __init__(self, all_documents, folds, num_test_documents, num_val_documents):

        assert (type(all_documents) is list)
        self.training_files_folds = []
        self.test_files_folds = []
        self.validation_files_folds = []
        self.folds = (len(all_documents)) / num_test_documents

        assert(self.folds == int(self.folds))
        
        for idx_fold in range(self.folds):
            
            idx_test_initial = int(idx_fold * num_test_documents)
            idx_test_end = int(idx_test_initial + num_test_documents)
            
            self.test_files_folds.append(all_documents[:][idx_test_initial:idx_test_end])
            
            
            idx_val_initial = idx_test_end % len(all_documents)
            idx_val_end = (idx_val_initial + num_val_documents)
            self.validation_files_folds.append(all_documents[:][idx_val_initial:idx_val_end])
            
            training_aux = []
            for idx_file in range(len(all_documents)):
                
                if idx_file not in range(idx_test_initial, idx_test_end) and idx_file not in range(idx_val_initial, idx_val_end):
                    training_aux.append(all_documents[:][idx_file])
                        
            self.training_files_folds.append(training_aux)

        self._i_integrity()





