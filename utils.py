import numpy as np
import glob
import json

np.random.seed(0)

img_paths = np.array(glob.glob('../Dataset/RSNA_2000/train_images/*dcm'))[:100]

permutation = np.random.choice(len(img_paths),len(img_paths),replace=False).astype(np.int32)

img_paths = img_paths[permutation].tolist()

# train val test split 0.7 0.1 0.2
data_split = {'train_imgs':img_paths[:int(0.7*len(img_paths))],
              'val_imgs':img_paths[int(0.7*len(img_paths)):int(0.8*len(img_paths))],
              'test_imgs':img_paths[int(0.8*len(img_paths)):],}



with open('data_split.json', 'w') as f:
    json.dump(data_split, f)