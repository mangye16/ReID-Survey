import os 
import random
import string

from utils import create_dir

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


src_dir_path = './mock_Dataset_without_eth'

gender_names = ['male', 'female']
age_names =['children', 'teenagers', 'young-adults', 'adults', 'seniors']

num_id = 1000

for i in range(1, num_id + 1):
    dir_path = src_dir_path + '/' + str(i)
    create_dir(dir_path)

    for j in range(random.randint(5, 50)):
        file_name = gender_names[random.randint(0, len(gender_names)-1)] + '_' + \
            age_names[random.randint(0, len(age_names)-1)] + '_' + \
            'cam-' + str(random.randint(1, 100)) + '_' + \
            '[date]_' + \
            get_random_string(8) + '.jpg'
        with open(dir_path + '/' + file_name, 'w') as f:
            pass

