import json
import shutil
from read_XML import *
import os
import random
from tqdm import tqdm

def split_dataset(json_path, source_folder):
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    train_folder = os.path.join(source_folder, 'train')
    val_folder = os.path.join(source_folder, 'val')
    test_folder = os.path.join(source_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Move files based on JSON list
    for set_type, file_list in data.items():
        target_folder = None

        if set_type == 'train':
            target_folder = train_folder
        elif set_type == 'val':
            target_folder = val_folder
        else:
            target_folder = test_folder

        for xml_file in tqdm(file_list, desc=f"Moving {set_type} files"):
            # Determine paths for XML and corresponding image file
            xml_source = os.path.join(source_folder, xml_file)
            img_source = xml_source.replace(".xml", ".jpg")
            
            if os.path.exists(xml_source) and os.path.exists(img_source):
                shutil.move(xml_source, os.path.join(target_folder, xml_file))
                shutil.move(img_source, os.path.join(target_folder, os.path.basename(img_source)))
            else:
                print(f"warning: {xml_source} or {img_source} not found.")





def train_val_split(json_path, source_folder):

    with open(json_path, 'r') as f:
        data = json.load(f)

    train_folder = os.path.join(source_folder, 'train')
    val_folder = os.path.join(source_folder, 'val')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Read the json file and extract the train image names
    train_images = data['train']
    val_images = data['val']

    # # Label random 10% of the train images as validation images
    # val_images = random.sample(train_images, int(0.1 * len(train_images)))
    # train_images = [img for img in train_images if img not in val_images]

    # # Update the json file with the new train and val image names
    # data['train'] = train_images
    # data['val'] = val_images
    # print(f"Updated train and val split. Train: {len(train_images)}, Val: {len(val_images)}")

    # # Write the updated json file
    # with open(json_path, 'w') as f:
    #     json.dump(data, f)

    # Move the validation images to the val folder
    for img in tqdm(val_images, desc='Moving validation images'):
        img_path = os.path.join(source_folder, img)

        print(img_path)

        shutil.move(img_path, os.path.join(val_folder, img))
        shutil.move(img_path.replace(".xml", ".jpg"), os.path.join(val_folder, img.replace(".xml", ".jpg")))
        
        # Move the obect proposal folders
        prop_folder = img_path.replace(".xml", "/")
        if os.path.exists(prop_folder):
            shutil.move(prop_folder, os.path.join(val_folder, img.replace(".xml", "/")))
        
    print(f"Moved {len(val_images)} images to the validation folder.")


# script_dir = os.path.dirname(os.path.abspath(__file__))
# source_folder = os.path.join(script_dir, 'Potholes', 'annotated-images', 'train')
# json_path = os.path.join(script_dir, 'Potholes', 'splits.json')

# train_val_split(json_path, source_folder)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_folder = os.path.join(script_dir, 'Potholes', 'annotated-images')
    json_path = os.path.join(script_dir, 'Potholes', 'splits.json')
    split_dataset(json_path, source_folder)