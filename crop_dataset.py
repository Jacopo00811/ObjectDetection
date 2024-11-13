import shutil
from turtle import pos
from read_XML import *
from object_proposal import SelectiveSearch, EdgeDetection
from evaluate_proposal import compute_IoU
import xml.etree.ElementTree as ET
import os
from xml.dom import minidom
from tqdm import tqdm


def initialize_xml_file(output_dir, img_name, img_shape):
    # Set up root annotation element
    annotation = ET.Element("annotation")
    
    # Add folder, filename, and path
    ET.SubElement(annotation, "folder").text = output_dir
    ET.SubElement(annotation, "filename").text = img_name
    ET.SubElement(annotation, "path").text = os.path.join(output_dir, img_name)
    
    # Add source element with database child
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    
    # Add size element with width, height, and depth
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])  # Width of the image
    ET.SubElement(size, "height").text = str(img_shape[0])  # Height of the image
    ET.SubElement(size, "depth").text = str(img_shape[2]) if len(img_shape) > 2 else "1"  # Depth
    
    # Add segmented element
    ET.SubElement(annotation, "segmented").text = "0"
    
    # Save the XML tree as an empty structure to the file
    tree = ET.ElementTree(annotation)
    xml_file_path = os.path.join(output_dir, img_name.replace('.jpg', '.xml'))
    tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)

    return xml_file_path

def add_object_to_xml(xml_file_path, box, proposal_name, label="predicted_object"):
    # Load the existing XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Create the object element
    object_element = ET.Element("object-proposal")
    ET.SubElement(object_element, "name").text = label
    ET.SubElement(object_element, "proposal").text = proposal_name
    ET.SubElement(object_element, "pose").text = "Unspecified"
    ET.SubElement(object_element, "truncated").text = "0"
    ET.SubElement(object_element, "difficult").text = "0"
    
    # Add bounding box coordinates
    bndbox = ET.SubElement(object_element, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(box[0])
    ET.SubElement(bndbox, "ymin").text = str(box[1])
    ET.SubElement(bndbox, "xmax").text = str(box[0] + box[2])
    ET.SubElement(bndbox, "ymax").text = str(box[1] + box[3])
    
    # Append object to root and save the updated XML file
    root.append(object_element)
    tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)



def save_proposals(images, annotations, names, output_dir, num_proposals, iou_threshold, algorithm='SelectiveSearch'):
    for i, (image, gt_boxes) in enumerate(tqdm(zip(images, annotations), total=len(images), desc="processing images")):
        if algorithm == 'SelectiveSearch':
            proposals = SelectiveSearch(image, num_proposals, mode='quality')
        elif algorithm == 'EdgeDetection':
            proposals = EdgeDetection(image, num_proposals)[0]
        else:
            raise ValueError("Invalid algorithm. Choose from 'SelectiveSearch' or 'EdgeDetection'")
        
        img_dir = os.path.join(output_dir, str(names[i])[:-4])
        os.makedirs(img_dir, exist_ok=True)
        # Create xml file for the image proposals
        xml_file_path = initialize_xml_file(img_dir, names[i], image.shape)

        pos_dir = os.path.join(img_dir, 'positive')
        neg_dir = os.path.join(img_dir, 'background')
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        for j, proposal in enumerate(tqdm(proposals, total=len(proposals), desc="saving proposals")):
            x, y, w, h = proposal
            crop_img = image[y:y+h, x:x+w]
            max_iou = max(compute_IoU(proposal, gt_box) for gt_box in gt_boxes)
            label = 'positive' if max_iou >= iou_threshold else 'background'
            # TODO: Add coordinetaes to the xml file
            add_object_to_xml(xml_file_path, proposal, f'img-{i+1}_{j}.jpg', label)
            save_dir = pos_dir if label == 'positive' else neg_dir
            cv2.imwrite(os.path.join(save_dir, f'img-{i+1}_{j}.jpg'), crop_img)
        # Free up memory
        img = img_dir + '.jpg'
        xml = img_dir + '.xml'
        # os.remove(img) # TODO: Uncomment this line when ready to run the full dataset
        # Move the xml to the new directory?? 
        # shutil.move(xml, pos_dir)
    print(f"Saved cropped proposals in {output_dir}!")


# # Test on a subset of the data
# images = images[:1]
# annotations = annotations[:1]
# names = names[:1]


if __name__ == "__main__":
    # Get the directory of the script

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the train images and annotations
    folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'train')
    images, annotations, names = read_images_and_xml(folder_path)

    # Temporarily run on subset
    # images = images[:1]
    # annotations = annotations[:1]
    # names = names[:1]

    save_proposals(images, annotations, names, output_dir=folder_path, num_proposals=1000, iou_threshold=0.6, algorithm='SelectiveSearch')

    # Read the validation images and annotations
    folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'val')
    images, annotations, names = read_images_and_xml(folder_path)

    # Temporarily run on subset
    # images = images[:1]
    # annotations = annotations[:1]
    # names = names[:1]

    save_proposals(images, annotations, names, output_dir=folder_path, num_proposals=1000, iou_threshold=0.6, algorithm='SelectiveSearch')

    # Read the test images and annotations
    folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'test')
    images, annotations, names = read_images_and_xml(folder_path)

    # Temporarily run on subset
    # images = images[:1]
    # annotations = annotations[:1]
    # names = names[:1]

    save_proposals(images, annotations, names, output_dir=folder_path, num_proposals=1000, iou_threshold=0.6, algorithm='SelectiveSearch')

