import os
import random


directory = ''
new_name_format = '{:>4}_c{}s{}_{}'


for filename in os.listdir(directory):
    if filename.endswith(".jpg") and "_" in filename:
        id, eid = filename.split("_")[:2]

        new_name = new_name_format.format(id, random.randint(1, 6), random.randint(1, 6), eid)

        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)