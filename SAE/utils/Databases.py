
import os, sys


DATABASE_LIST = [
                    'Einsiedeln', 
                    'Salzinnes', 
                    'Dibco', 
                    'Dibco14', 
                    'Dibco16', 
                    'Palm', 
                    'Palm1', 
                    'Palm2', 
                    'PHI', 
                    'Einsiedeln_inv', 
                    'Salzinnes_inv', 
                    'Dibco_inv', 
                    'Dibco14_inv', 
                    'Dibco16_inv', 
                    'PHI_inv',
                    'Palm_inv']
MURET_DATABASE_LIST = ['b-3-28', 'b-50-747', 'b-53-781', 'b-59-850', 'SEILS', 'testing_images']


def getFullPathParentDatabasesContainer():
    return os.path.dirname(os.path.abspath(__file__)) + "/" + "databases/datasets_bin/"

def getFullPathParentDatabasesContainerMURET():
    return os.path.dirname(os.path.abspath(__file__)).replace("/utils", "") + "/" + "databases/MURET/"

def getPathdirDatabase(db_name):
    pathdir_exec = getFullPathParentDatabasesContainer()

    db_dir_name = db_name
    if "Dibco" in db_name:
        db_dir_name = "Dibco"
    elif "Palm" in db_name:
        db_dir_name = "Palm"

    pathdir_src_files = pathdir_exec + "SRC/" + db_dir_name
    pathdir_gt_files = pathdir_exec  + "GT/" + db_dir_name

    return [pathdir_src_files, pathdir_gt_files]


def getPathdirDatabaseSRCAndJSON_MURET(db_name, with_gt=False):
    pathdir_exec = getFullPathParentDatabasesContainerMURET()

    if type(db_name) is list:
        pathdir_src_files = []
        pathdir_gt_files = []
        pathdir_json_files = []

        for db_name_i in db_name:
            pathdir_src_files.append(pathdir_exec + "SRC/" + db_name_i)
            pathdir_json_files.append(pathdir_exec  + "JSON/" + db_name_i)
            if with_gt:
                pathdir_gt_files.append(pathdir_exec  + "GT/" + db_name_i)
    else:
        pathdir_src_files = pathdir_exec + "SRC/" + db_name
        pathdir_json_files = pathdir_exec  + "JSON/" + db_name
        if with_gt:
            pathdir_gt_files = pathdir_exec  + "GT/" + db_name

    if with_gt:
        return [pathdir_src_files, pathdir_gt_files, pathdir_json_files]
    else:
        return [pathdir_src_files, pathdir_json_files]