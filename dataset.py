from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from read_XML import from_indeces_to_coords
import os
from PIL import Image
import random
import torch

class CroppedProposalDataset(Dataset):
    def __init__(self, mode, transform, dir='Potholes/annotated-images', size=256):
        path_list = dir.split('/')
        dir = os.path.join(os.getcwd(), *path_list)
        self.transform = transform
        self.size = size
        # self.skip = ["img-304", "img-355", "img-592", "img-598"]
        if mode == 'train':
            self.dir = os.path.join(dir, 'train')
        elif mode == 'val':
            self.dir = os.path.join(dir, 'val')
        elif mode == 'test':
            self.dir = os.path.join(dir, 'test')
        else:
            raise ValueError("\nInvalid mode. Please choose 'train', 'val', or 'test'\n")


        self.img_dirs = [os.path.join(self.dir, f) 
                            for f in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, f))] # Collects the list of image directories 
        self.list_of_images = [] 
        self.list_of_labels = []
        self.list_of_xml_dir = []
        
        for image_dir in self.img_dirs:
            self.pos_dir = os.path.join(image_dir, 'positive')
            self.neg_dir = os.path.join(image_dir, 'background')
            self.xml_dir = os.path.join(image_dir, image_dir+".xml")

            self.pos_images = [os.path.join(self.pos_dir, f) for f in os.listdir(self.pos_dir)]
            self.neg_images = [os.path.join(self.neg_dir, f) for f in os.listdir(self.neg_dir)]
            
            if mode == 'test':
                # No upsampling for test data
                number = len(self.pos_images)
                self.neg_images = self.neg_images[:number*3]
                self.images = self.pos_images + self.neg_images
                self.labels = [1] * len(self.pos_images) + [0] * len(self.neg_images)
                self.list_of_images.append(self.images)
                self.list_of_labels.append(self.labels)
            else:
                # Shuffle negatives 
                random.shuffle(self.neg_images)
                self.neg_images = self.neg_images[:24]

                self.images, self.labels = self.upsample_if_needed(self.transform, self.pos_images, self.neg_images)
                self.list_of_images.append(self.images)
                self.list_of_labels.append(self.labels)
            self.list_of_xml_dir.append(self.xml_dir)
        
        # Remove empty lists if any
        self.list_of_images = [lst for lst in self.list_of_images if lst]
        self.list_of_labels = [lst for lst in self.list_of_labels if lst]
        self.list_of_xml_dir = [lst for lst in self.list_of_xml_dir if lst]
        
        # Transforms applied before output 
        self.transform =  transforms.Compose([ 
            transforms.Resize((self.size, self.size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)])

    def upsample_if_needed(self, transform, pos_images, neg_images):
        upsampled_pos_images = []
        if len(pos_images) == 0:
            return [], []
        # If there are fewer than 8 positive images, apply transformations to reach 8 images
        if len(pos_images) > 0 and len(pos_images) < 8:
            upsampled_pos_images.extend(pos_images)
            # Apply transformations to positive images until we reach 8 in total
            while len(upsampled_pos_images) < 8:
                for img in pos_images:
                    if len(upsampled_pos_images) < 8:
                        upsampled_pos_images.append(transform(img))
                    else:
                        break
        else:
            upsampled_pos_images = pos_images[:8]  # If 8 or more positives get the first 8 images

        images = upsampled_pos_images + neg_images        
        labels = [1] * len(upsampled_pos_images) + [0] * len(neg_images)

        return images, labels

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        # Get the entire list of proposals and labels for a single image directory
        image_paths = self.list_of_images[idx]  # List of paths for proposals
        labels = self.list_of_labels[idx]  # List of corresponding labels
        xml_dir = self.list_of_xml_dir[idx] # .xml file's name

        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined)
        # Get the unique indeces for each proposal (index of specific proposal give a certain image)
        unique_indeces = [int(path.split('/')[-1].split('.')[0].split('_')[1]) for path in image_paths]
        # print(unique_indeces)
        # print("XML DIR:", xml_dir)
        coords = from_indeces_to_coords(unique_indeces, xml_dir, labels)
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        
        # Apply the output transformations to each image
        if self.transform:
            images = [self.transform(image) for image in images]

        return torch.stack(images), torch.tensor(labels), xml_dir, coords

transform = transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, saturation=0.3),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomAdjustSharpness(2, p=0.5),
])


################################# TESTING ##################################
# /zhome/f9/0/168881/

# dir='/dtu/blackhole/11/168881/ObjectDetection/Potholes/annotated-images'
# train_dataset = CroppedProposalDataset('val', transform=transform, size=256)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) # !! Do not shuffle here and do not change batch_size !!

# import matplotlib.pyplot as plt

# for batch_idx, (images, labels, xml_dir, coords) in enumerate(train_loader):
#     images = images.squeeze(0)  # Remove the batch dimension when batch_size is 1
#     labels = labels.squeeze(0)  # Remove the batch dimension when batch_size is 1
#     xml_dir = xml_dir[0]  # Remove the tuple
#     # print("XML in batch:", xml_dir)
#     print(f"Batch {batch_idx + 1}:")
#     print(f"  Number of images in batch: {len(images)}")  
#     print(f"  Number of labels in batch: {len(labels)}")
#     print(f"  Labels: {labels}")
#     print(f"  Coords: {coords}")
#     print(f"  Shape of images: {images.shape}")
#     print(f"  XML file: {xml_dir}")
#     image_name = xml_dir.split('\\')[-1].split('.')[0] + '.jpg'
#     print(f"  Image name: {image_name}")

    # print(f"Example image: {images[0]}")

    # Plot images in the first batch
    # if batch_idx == 6:

    #     plt.figure(figsize=(20, 20))
    #     for i in range(len(images)):
    #         img = images[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for plotting
    #         plt.subplot(4, 8, i + 1)
    #         plt.imshow(img)
    #         plt.axis('off')

    #         # Add label text on the image
    #         label = labels[i].item()  # Convert label tensor to a scalar
    #         plt.text(5, 5, f"Label: {label}", color='white', fontsize=12, ha='left', va='top', backgroundcolor='black')
    #     plt.savefig('batch_with_labels.png')
        

    #     print(os.path.join("Potholes", "annotated-images", "train", image_name))
    #     image = Image.open(os.path.join("Potholes", "annotated-images", "train", image_name))
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.title("Original Image")
    #     plt.savefig('original_image.png')
