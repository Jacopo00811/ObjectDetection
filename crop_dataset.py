import shutil
from turtle import pos
from read_XML import *
from object_proposal import SelectiveSearch, EdgeDetection
from evaluate_proposal import compute_IoU

def save_proposals(images, annotations, names, output_dir, num_proposals, iou_threshold, algorithm='SelectiveSearch'):
    for i, (image, gt_boxes) in enumerate(zip(images, annotations)):
        if algorithm == 'SelectiveSearch':
            proposals = SelectiveSearch(image, num_proposals, mode='quality')
        elif algorithm == 'EdgeDetection':
            proposals = EdgeDetection(image, num_proposals)[0]
        else:
            raise ValueError("Invalid algorithm. Choose from 'SelectiveSearch' or 'EdgeDetection'")
        
        img_dir = os.path.join(output_dir, str(names[i])[:-4])
        os.makedirs(img_dir, exist_ok=True)
        pos_dir = os.path.join(img_dir, 'positive')
        neg_dir = os.path.join(img_dir, 'background')
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        for j, proposal in enumerate(proposals):
            x, y, w, h = proposal
            crop_img = image[y:y+h, x:x+w]
            max_iou = max(compute_IoU(proposal, gt_box) for gt_box in gt_boxes)
            label = 'positive' if max_iou >= iou_threshold else 'background'
            
            save_dir = pos_dir if label == 'positive' else neg_dir
            cv2.imwrite(os.path.join(save_dir, f'img-{i+1}_{j}.jpg'), crop_img)
        # Free up memory
        img = img_dir + '.jpg'
        xml = img_dir + '.xml'
        # os.remove(img) # TODO: Uncomment this line when ready to run the full dataset
        # Move the xml to the new directory?? 
        # shutil.move(xml, pos_dir)
    print(f"Saved cropped proposals in {output_dir}!")


# script_dir = os.path.dirname(os.path.abspath(__file__))
# folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'train')
# images, annotations, names = read_images_and_xml(folder_path)

# # Test on a subset of the data
# images = images[:1]
# annotations = annotations[:1]
# names = names[:1]
# save_proposals(images, annotations, names, output_dir=folder_path, num_proposals=1000, iou_threshold=0.6, algorithm='SelectiveSearch')
