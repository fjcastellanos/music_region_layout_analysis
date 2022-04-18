
try:
    from utils.file_manager import FileManager
except:
    from file_manager import FileManager

def invertColorsDatabase(path_src_original, path_gt_original, db_name):
    assert(type(path_src_original) is str)
    assert(type(path_gt_original) is str)


    list_pathfiles_src_original = FileManager.listFilesRecursive(path_src_original)
    list_pathfiles_gt_original = FileManager.listFilesRecursive(path_gt_original)
    assert(len(list_pathfiles_gt_original) == len(list_pathfiles_gt_original))

    for i in range(len(list_pathfiles_gt_original)):
        pathfile_src_original = list_pathfiles_src_original[i]
        pathfile_gt_original = list_pathfiles_gt_original[i]

        im_src = FileManager.loadImage(pathfile_src_original, True)
        im_gt = FileManager.loadImage(pathfile_gt_original, False)

        pathfile_src_modified = pathfile_src_original
        pathfile_gt_modified = pathfile_gt_original

        pathfile_src_modified = pathfile_src_modified.replace("/"+db_name+"/", "/" + db_name + "_inv/")
        pathfile_gt_modified =  pathfile_gt_modified.replace("/"+db_name+"/", "/" + db_name + "_inv/")

        im_src = 255 - im_src

        FileManager.saveImageFullPath(im_src, pathfile_src_modified)
        FileManager.saveImageFullPath(im_gt, pathfile_gt_modified)



if __name__ == '__main__':

    print("Database Modifier")

    try:
        from utils.Databases import DATABASE_LIST, getPathdirDatabase
    except:
        from Databases import DATABASE_LIST, getPathdirDatabase


    for db_name in DATABASE_LIST:

        if "_inv" not in db_name:
            [pathdir_src_files, pathdir_gt_files] = getPathdirDatabase(db_name)
            invertColorsDatabase(pathdir_src_files, pathdir_gt_files, db_name)

    
