import json
import shutil
from read_XML import *

def split_dataset(json_path, source_folder):
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    train_folder = os.path.join(source_folder, 'train')
    test_folder = os.path.join(source_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Move files based on JSON list
    for set_type, file_list in data.items():
        target_folder = train_folder if set_type == 'train' else test_folder

        for xml_file in file_list:
            # Determine paths for XML and corresponding image file
            xml_source = os.path.join(source_folder, xml_file)
            img_source = xml_source.replace(".xml", ".jpg")
            
            if os.path.exists(xml_source) and os.path.exists(img_source):
                shutil.move(xml_source, os.path.join(target_folder, xml_file))
                shutil.move(img_source, os.path.join(target_folder, os.path.basename(img_source)))
            else:
                print(f"warning: {xml_source} or {img_source} not found.")



# script_dir = os.path.dirname(os.path.abspath(__file__))
# source_folder = os.path.join(script_dir, 'Temp', 'annotated-images')
# json_path = os.path.join(script_dir, 'Temp', 'splits.json')
# split_dataset(json_path, source_folder)
