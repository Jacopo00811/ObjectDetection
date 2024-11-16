import os
import cv2
import xml.etree.ElementTree as ET
import torch

# script_dir = os.path.dirname(os.path.abspath(__file__))
# folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'train')
def read_xml_gt(xml_dir):

    tree = ET.parse(xml_dir)
    root = tree.getroot()

    annotations = []
    for obj in root.findall('object-proposal'):
        annotation = {
            'name': obj.find('name').text,
            'proposal': obj.find('proposal').text,
            'pose': obj.find('pose').text,
            'truncated': int(obj.find('truncated').text),
            'difficult': int(obj.find('difficult').text),
            'bndbox': {
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
        }
        annotations.append(annotation)

    coords = [[]]
    for annotation in annotations:
        coords.append([annotation['bndbox']['xmin'], annotation['bndbox']['ymin'], annotation['bndbox']['xmax'], annotation['bndbox']['ymax']])
    
    return torch.tensor(coords)

def read_xml_gt_og(xml_dir):
    tree = ET.parse(xml_dir)
    root = tree.getroot()

    annotations = []
    for obj in root.findall('object'):
        annotation = {
            'name': obj.find('name').text,
            'pose': obj.find('pose').text,
            'truncated': int(obj.find('truncated').text),
            'difficult': int(obj.find('difficult').text),
            'bndbox': {
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
        }

        annotations.append(annotation)

    coords = []
    for annotation in annotations:
        # print(f'here')
        coords.append([annotation['bndbox']['xmin'], 
                       annotation['bndbox']['ymin'], 
                       annotation['bndbox']['xmax'], 
                       annotation['bndbox']['ymax']])
    
    coords = torch.tensor(coords)    
    coords = coords.squeeze()
    
    return coords



def read_images_and_xml(folder_path):
    images = []
    annotations = []
    names = []

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

                # Extract annotations
                img_annotations = []
                name = root.find('filename').text
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    # name = obj.find('name').text
                    # Calculate width and height
                    width = xmax - xmin
                    height = ymax - ymin

                    # Append to annotations list
                    img_annotations.append([xmin, ymin, width, height])
                names.append(name)
                annotations.append(img_annotations)

    return images, annotations, names


def from_indeces_to_coords(list_of_indeces, xml_dir, labels):
    """ Returns a matrix of coordinates of 32x6 elemnts: xmin, ymin, xmax, ymax, label, unique, index for each batch of 32 images """
    
    coords = torch.zeros(len(list_of_indeces), 6, dtype=torch.int32)
    image_number = xml_dir.split('/')[-1].split('-')[-1].split('.')[0]
    # New path after the split of dataset
    dir = os.path.join(os.path.dirname(xml_dir), f'img-{image_number}', f'img-{image_number}.xml')   
    # print("DIr:", dir) 
    tree = ET.parse(dir)
    root = tree.getroot()

    annotations = []
    for obj in root.findall('object-proposal'):
        annotation = {
            'name': obj.find('name').text,
            'proposal': obj.find('proposal').text,
            'pose': obj.find('pose').text,
            'truncated': int(obj.find('truncated').text),
            'difficult': int(obj.find('difficult').text),
            'bndbox': {
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
        }
        annotations.append(annotation)
    # print(annotations)
    for i, index in enumerate(list_of_indeces):
        coords[i] = torch.tensor([annotations[index]['bndbox']['xmin'],
                                  annotations[index]['bndbox']['ymin'],
                                  annotations[index]['bndbox']['xmax'],
                                  annotations[index]['bndbox']['ymax'],
                                  labels[i],
                                  index], dtype=torch.int
                                )

    return coords


def draw_annotations(images, annotations, image_index=0):
    # Draw annotations on the first image
    for annotation in annotations[image_index]:
        xmin, ymin, width, height = annotation
        xmax = xmin + width
        ymax = ymin + height

        # Draw bounding box
        cv2.rectangle(images[image_index], (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put label
        # cv2.putText(images[image_index], name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # Save the modified image to a file
    output_path = os.path.join(os.path.dirname(__file__), f"annotated_image_{image_index}.jpg")
    cv2.imwrite(output_path, images[image_index])
    print(f"Image saved at {output_path}")



# images, annotations, names = read_images_and_xml(folder_path)
# print(f"Found {len(images)} images and {len(annotations)} annotations")
# draw_annotations(images, annotations, image_index=3)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = "Potholes/annotated-images/train/img-665.xml"
    full_path = os.path.join(script_dir, xml_path)

    coords = read_xml_gt_og(full_path)
    print(coords)
    