import os
import cv2
import xml.etree.ElementTree as ET

def read_images_and_xml(folder_path):
    images = []
    annotations = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            xml_path = os.path.join(folder_path, filename.replace(".jpg", ".xml"))

            if os.path.exists(xml_path):
                # Read image
                image = cv2.imread(image_path)
                images.append(image)

                # Read XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                annotations.append(root)

    return images, annotations


def draw_annotations(images, annotations):
    # Draw annotations on the first image
    for annotation in annotations[0].findall('object'):
        name = annotation.find('name').text
        bndbox = annotation.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Draw bounding box
        cv2.rectangle(images[0], (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put label
        cv2.putText(images[0], name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Find the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the current folder path
folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images')
images, annotations = read_images_and_xml(folder_path)

