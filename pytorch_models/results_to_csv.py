import numpy as np
import pandas as pd
import sys
import pandas as pd


# Constants
COLUMNS_DATA_AUG = [
      np.array(['MODEL', 'DATASET', 'PAGES', 'IMAGES', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS']),
      np.array(['', '', '', '', 'MAP_COCO', 'MAP_COCO_STAFF', 'MAP_COCO_LYRICS', 'MAP_VOC', 'MAP_VOC_STAFF', 'MAP_VOC_LYRICS', 'MAP_STRICT', 'MAP_STRICT_STAFF', 'MAP_STRICT_LYRICS'])
]

COLUMNS_NO_AUG = [
      np.array(['MODEL', 'DATASET', 'IMAGES', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS']),
      np.array(['', '', '', 'MAP_COCO', 'MAP_COCO_STAFF', 'MAP_COCO_LYRICS', 'MAP_VOC', 'MAP_VOC_STAFF', 'MAP_VOC_LYRICS', 'MAP_STRICT', 'MAP_STRICT_STAFF', 'MAP_STRICT_LYRICS'])
]

N_PAGES=[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]
N_IMAGES=[100]#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def parse_results(results):
  results = [float(res[res.find("D")+2:]) for res in results]

  return results

def get_results(results, num_results, total_results):
  for n_results in range(0, total_results, num_results):
    yield np.array(results[n_results:(n_results+num_results)], dtype=np.float32)


def results_to_csv(arch, dataset, data_aug, file_in, file_out):
  COLUMNS = COLUMNS_DATA_AUG if data_aug!=0 else COLUMNS_NO_AUG
  print(f'DATA AUGMENTATION: -{data_aug}-')

  with open(file_in) as file_results:
    # Read results from a .txt file
    RESULTS = [result.rstrip() for result in file_results]

    # Parse results in order to get only the results
    RESULTS = parse_results(RESULTS)
    print(f'Len results: {len(RESULTS)}')
    len_res = len(RESULTS)

    # Create DataFrame
    df = pd.DataFrame(columns=COLUMNS)
    # results_df = RESULTS
    generator = get_results(RESULTS, 9, len_res) # generator

    if data_aug != 0: # DATA AUGMENTATION
      for pages in N_PAGES: 
        for images in N_IMAGES:
          data_df = [arch, dataset, pages, images]
          results_yielded = next(generator)
          concat = np.concatenate((data_df, results_yielded))
          df_add = pd.DataFrame(np.expand_dims(np.array(concat), axis=0), columns=COLUMNS)
          df = df.append(df_add)

    else: # WITHOUT DATA AUGMENTATION
      for images in N_PAGES:
        data_df = [arch, dataset, images]
        results_yielded = next(generator)
        concat = np.concatenate((data_df, results_yielded))
        df_add = pd.DataFrame(np.expand_dims(np.array(concat), axis=0), columns=COLUMNS)
        df = df.append(df_add)
    
    # Convert results to float types
    df['RESULTS'] = df['RESULTS'].astype(np.float32)
    print(df.dtypes)

    # Export to a .csv file
    df.to_csv(r"./"+file_out, index=False, decimal=',')


if __name__ == "__main__":
    if len(sys.argv)!= 6:
        raise Exception('Program requires 5 arguments!')
    else:
        arch = sys.argv[1]
        dataset = sys.argv[2]
        data_aug = int(sys.argv[3])
        file_in = sys.argv[4]
        file_out = sys.argv[5]
        results_to_csv(arch, dataset, data_aug, file_in, file_out)
