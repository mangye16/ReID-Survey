import os 


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)