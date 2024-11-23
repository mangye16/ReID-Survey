import os
import random
import shutil
from collections import defaultdict

def split_dataset_by_id(dataset_folder, train_folder, test_folder, train_ratio=0.7):
  
    image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.jpg')]

 
    id_groups = defaultdict(list)
    for image_file in image_files:
        id_part = image_file.split('_')[0]
        id_groups[id_part].append(image_file)

   
    id_list = list(id_groups.keys())
    random.shuffle(id_list)
    split_index = int(len(id_list) * train_ratio)
    train_ids = id_list[:split_index]
    test_ids = id_list[split_index:]

   
    train_images = set()
    test_images = set()

    for id_part in train_ids:
        train_images.update(id_groups[id_part])

    for id_part in test_ids:
        test_images.update(id_groups[id_part])

  
    for image_file in train_images:
        source_path = os.path.join(dataset_folder, image_file)
        destination_path = os.path.join(train_folder, image_file)
        shutil.move(source_path, destination_path)


    for image_file in test_images:
        source_path = os.path.join(dataset_folder, image_file)
        destination_path = os.path.join(test_folder, image_file)
        shutil.move(source_path, destination_path)

if __name__ == "__main__":
    dataset_folder = "dataset"  # dataset_folder
    train_folder = "train"  # train_folder
    test_folder = "test"  # test_folder

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    train_ratio = 0.7 # train_set ratio
    split_dataset_by_id(dataset_folder, train_folder, test_folder, train_ratio)