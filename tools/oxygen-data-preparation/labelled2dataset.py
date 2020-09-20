import os 
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import random 

from utils import create_dir

seed = 1024
np.random.seed(seed)
random.seed(seed)

# Dataset_without_eth 
#   - date
#        [N] [gender]_[age]_[id]/[gender]_[age]_[camid]_[date_timestamp]_[anonymous].jpg
#        [Y] [id]/[gender]_[age]_[camid]_[date_timestamp]_[anonymous].jpg


# labelled dataset to labelled-pool

src_dir_path = './mock_Dataset_without_eth'
dst_dir_path = './Oxygen1'
train_dir_path = dst_dir_path + '/bounding_box_train'
create_dir(train_dir_path)
query_dir_path = dst_dir_path + '/query'
create_dir(query_dir_path)
gallery_dir_path = dst_dir_path + '/bounding_box_test'
create_dir(gallery_dir_path)
hasDistractor = True
if hasDistractor:
    distractor_dir_path = dst_dir_path + '/distractor'
    create_dir(distractor_dir_path)
num_distractor = 10

org_img_paths = list()
processed_file_names = list()
path2newname_mapper = dict()

num_people = len(os.listdir(src_dir_path))
distractor_pids = list(range(1, num_people+1))
random.shuffle(distractor_pids)
distractor_pids = distractor_pids[:num_distractor]
distractor_uniques = [e-1 for e in sorted(distractor_pids)]
distractor_pairs = list()

for root, subdirs, file_names in os.walk(src_dir_path):
    if len(file_names) != 0:
        pid = root[root.rfind('/')+1:]
        for file_name in file_names:
            if file_name[0] == '.': continue
            gender, age, camid, timestamp, uniqueid = file_name.split('_')                
            # ! uniqueid (unique_id + '.jpg')
            if hasDistractor: 
                if int(pid) in distractor_pids:
                    distractor_pairs.append([root + '/' + file_name, pid + '_' + camid + '_' + uniqueid])
                else:
                    path2newname_mapper[root + '/' + file_name] = pid + '_' + camid + '_' + uniqueid
            else:
                path2newname_mapper[root + '/' + file_name] = pid + '_' + camid + '_' + uniqueid
                
# split train, test, distractor 
# stratify between train and test
labels = [int(file_name.split('_')[0]) - 1  for file_name in path2newname_mapper.values()]

print('The number of ids : ', num_people)
print(len(list(path2newname_mapper.keys())))
print('-' * 30)
print(len(labels))

pairs = np.array([[img_path, file_name] for img_path, file_name in path2newname_mapper.items()])
train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, stratify=labels, test_size=0.64)
gallery_pairs, query_pairs, gallery_labels, query_labels = train_test_split(test_pairs, test_labels, stratify=test_labels, test_size=0.15 )

train_uniques = np.unique(np.array(train_labels))
gallery_uniques = np.unique(np.array(gallery_labels))
query_uniques = np.unique(np.array(gallery_labels))
print('train unqiue id :', len(train_uniques))
print('gallery unqiue id :', len(gallery_uniques))
print('query unique id : ', len(query_uniques))

with open(dst_dir_path + '/' + 'train_uniques.txt', 'w') as f:
    for i in list(train_uniques):
        f.write(str(i) + '\n')
with open(dst_dir_path + '/' + 'gallery_uniques.txt', 'w') as f:
    for i in list(gallery_uniques):
        f.write(str(i) + '\n')
with open(dst_dir_path + '/' + 'query_uniques.txt', 'w') as f:
    for i in list(query_uniques):
        f.write(str(i) + '\n')
if hasDistractor:
    with open(dst_dir_path + '/' + 'distractor_uniques.txt', 'w') as f:
        for i in distractor_uniques:
            f.write(str(i) + '\n')

for img_path, file_name in tqdm(train_pairs):
    shutil.copyfile(img_path, train_dir_path + '/' + file_name)
for img_path, file_name in tqdm(gallery_pairs):
    shutil.copyfile(img_path, gallery_dir_path + '/' + file_name)
for img_path, file_name in tqdm(query_pairs):
    shutil.copyfile(img_path, query_dir_path + '/' + file_name)
if hasDistractor:
    for img_path, file_name in tqdm(distractor_pairs):
        shutil.copyfile(img_path, distractor_dir_path + '/' + file_name)
