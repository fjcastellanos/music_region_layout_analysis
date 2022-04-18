
from Label import Label
from file_manager import FileManager


class GTDocument:

    labels = None
    path_file = None

    def _i_integrity(self):
        assert(type(self.labels) is dict)
        assert(type(self.path_file) is str)
        assert(self.path_file != "")

        for label in self.labels:
            assert(isinstance(label, Label))



    def __init__(self, labels, path_file):
        self._i_integrity()

        self.labels = labels
        self.path_file = path_file

    def _i_with_exact_labels(self, labels):
        self._i_integrity()
        assert(type(labels) is list)

        if labels == self.labels:
            return True
        else:
            return False


    def with_exact_labels(self, labels):
        return self._i_with_exact_labels(labels)
        

    def get_gt_image_with_exact_labels(self, labels):
        self._i_integrity()
        assert(self.with_exact_labels(labels))
        return FileManager.loadImage(self.path_file, False)


