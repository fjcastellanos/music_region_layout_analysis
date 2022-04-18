class Document:
    def __init__(self, src_pathfile, gt_pathfile, json_pathfile=None):
        self.src_pathfile = src_pathfile
        self.gt_pathfile = gt_pathfile
        self.json_pathfile = json_pathfile

    def __repr__(self):
        if self.json_pathfile is not None:
            str_json_pathfile = "\n" + self.json_pathfile
        else:
            str_json_pathfile = ""
        return self.src_pathfile + "\n" + self.gt_pathfile + str_json_pathfile

    def __str__(self):
        return self.__repr__()