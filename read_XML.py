import os
import cv2
import xml.etree.ElementTree as ET

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images')

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


def draw_annotations(images, annotations, image_index=0):
    # Draw annotations on the first image
    for annotation in annotations[image_index].findall('object'):
        name = annotation.find('name').text
        bndbox = annotation.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Draw bounding box
        cv2.rectangle(images[image_index], (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put label
        cv2.putText(images[image_index], name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # Save the modified image to a file
    output_path = os.path.join(os.path.dirname(__file__), f"annotated_image_{image_index}.jpg")
    cv2.imwrite(output_path, images[image_index])
    print(f"Image saved at {output_path}")



# images, annotations = read_images_and_xml(folder_path)
# print(f"Found {len(images)} images and {len(annotations)} annotations")
# draw_annotations(images, annotations, image_index=3)

