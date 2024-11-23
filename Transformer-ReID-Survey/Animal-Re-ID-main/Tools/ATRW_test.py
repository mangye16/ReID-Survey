import os
import json
import shutil

json_file_path = "gt_test_plain.json"
image_folder_path = ""
output_folder_path = ""

os.makedirs(output_folder_path, exist_ok=True)

with open(json_file_path, "r") as file:
    data = json.load(file)

    for item in data:
        entityid = int(item["entityid"])
        imgid = int(item["imgid"])
        query = item["query"]
        if query == "multi":
            img_files = [file_name for file_name in os.listdir(image_folder_path) if file_name.endswith(".jpg")]
            matching_files = [file_name for file_name in img_files if int(os.path.splitext(file_name)[0]) == imgid]

            if len(matching_files) > 0:
                old_file_name = os.path.join(image_folder_path, matching_files[0])
                new_file_name = os.path.join(output_folder_path, f"{entityid}_{imgid}.jpg")

                shutil.copy2(old_file_name, new_file_name)