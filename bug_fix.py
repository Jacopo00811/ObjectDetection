import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


def fix_img_naming(folder_path):


    # Get the list of folders in the folder
    dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # Loop through the directories
    for dir in tqdm(dirs, desc='Processing folders'):
        # get the name of the folder
        folder_name = os.path.basename(dir)
        
        # find the xml within the folder
        xml_file = folder_name + '.xml'
        
        # read the xml file
        tree = ET.parse(os.path.join(folder_path, dir, xml_file))

        # get the root of the xml
        root = tree.getroot()
        
        # get the object proposals
        objects = root.findall('object-proposal')
        
        # loop through the object proposals
        for obj in objects:
            # get the name of the object
            name = obj.find('proposal').text

            proposal_number = name.split('_')[1]

            # replace proposal name
            proposal_name = folder_name + '_' + proposal_number


            # set the new proposal name
            obj.find('proposal').text = proposal_name
        
        # write the new xml file
        tree.write(os.path.join(folder_path, dir, xml_file), encoding="utf-8", xml_declaration=True)

        # update the proposal image name

        # Parse the image folder background dir
        img_folder = os.path.join(folder_path, dir, 'background')
        # Get the list of images in the folder
        imgs = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
        # Loop through the images
        for img in imgs:
            # Get the image name
            img_name = os.path.basename(img)
            # Get the proposal number
            proposal_number = img_name.split('_')[1]
            # Replace the proposal name
            new_img_name = folder_name + '_' + proposal_number
            # Rename the image
            os.rename(os.path.join(img_folder, img), os.path.join(img_folder, new_img_name))
        
        # Dothe same for the positive images
        img_folder = os.path.join(folder_path, dir, 'positive')
        # Get the list of images in the folder
        imgs = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
        # Loop through the images
        for img in imgs:
            # Get the image name
            img_name = os.path.basename(img)
            # Get the proposal number
            proposal_number = img_name.split('_')[1]
            # Replace the proposal name
            new_img_name = folder_name + '_' + proposal_number
            # Rename the image
            os.rename(os.path.join(img_folder, img), os.path.join(img_folder, new_img_name))



def assert_proposal_length(folder_path):

    # get the list of folders in the folder
    dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # loop through the directories
    for dir in tqdm(dirs, desc='Processing folders'):
        
        # get the folder name
        folder_name = os.path.basename(dir)

        # find the xml within the folder
        xml_file = folder_name + '.xml'

        # read the xml file
        tree = ET.parse(os.path.join(folder_path, dir, xml_file))

        # get the root of the xml
        root = tree.getroot()

        # get the object proposals
        objects = root.findall('object-proposal')

        # get the number of proposals
        num_proposals = len(objects)

        # Parse the image folder background dir
        img_folder = os.path.join(folder_path, dir, 'background')
        # Get the list of images in the folder
        imgs = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
        # Get the number of images
        num_imgs = len(imgs)
        
        # parse the image folder positive dir
        img_folder = os.path.join(folder_path, dir, 'positive')
        # Get the list of images in the folder
        imgs = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
        # Get the number of images
        num_imgs += len(imgs)

        # assert the number of proposals is equal to the number of images
        check = num_proposals == num_imgs == 1000

        if not check:
            print(f"For folder {folder_name} Number of proposals: {num_proposals}, Number of images: {num_imgs}")

            # save the filepath to the folder
            with open('bug_fix.txt', 'a') as f:
                f.write(f"{folder_name}\n")
        

def fix_again(folder_path):
    # Get the list of folders in the folder
    dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # Loop through the directories
    for dir in tqdm(dirs, desc='Processing folders'):
        # get the name of the folder
        folder_name = os.path.basename(dir)

        # Parse the image folder background dir
        img_folder = os.path.join(folder_path, dir, 'background')
        # Get the list of images in the folder
        imgs = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
        # Loop through the images
        for img in imgs:
            # Get the image name
            img_name = os.path.basename(img)
            # Get the proposal number
            proposal_number = img_name.split('_')[1]
            # Replace the proposal name
            new_img_name = folder_name + '_' + proposal_number
            # Rename the image
            os.rename(os.path.join(img_folder, img), os.path.join(img_folder, new_img_name))
        
        # Dothe same for the positive images
        img_folder = os.path.join(folder_path, dir, 'positive')
        # Get the list of images in the folder
        imgs = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
        # Loop through the images
        for img in imgs:
            # Get the image name
            img_name = os.path.basename(img)
            # Get the proposal number
            proposal_number = img_name.split('_')[1]
            # Replace the proposal name
            new_img_name = folder_name + '_' + proposal_number
            # Rename the image
            os.rename(os.path.join(img_folder, img), os.path.join(img_folder, new_img_name))

    
if __name__ == "__main__":

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the train images and annotations
    folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'train')
    # fix_img_naming(folder_path)
    assert_proposal_length(folder_path)


    # Read the validation images and annotations
    folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'val')
    # fix_img_naming(folder_path)
    assert_proposal_length(folder_path)

    # Read the test images and annotations
    folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'test')
    # fix_img_naming(folder_path)
    assert_proposal_length(folder_path)

