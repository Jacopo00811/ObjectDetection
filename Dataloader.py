from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class CroppedProposalDataset(Dataset):
    def __init__(self, mode, transform=None, dir='Potholes/annotated-images'):
        # split dir by / and join list with current working directory
        path_list = dir.split('/')
        dir = os.path.join(os.getcwd(), *path_list)

        if mode == 'train':
            self.dir = os.path.join(dir, 'train')
            self.img_dirs = [os.path.join(self.dir, f) 
                             for f in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, f)) # Collects the list of image directories 
                             ]
            self.pos_dirs = [os.path.join(d, 'positive') for d in self.img_dirs] # adds the positive directory to the list of image directories
            self.neg_dirs = [os.path.join(d, 'background') for d in self.img_dirs] # adds the background directory to the list of image directories
            # self.pos_dir = os.path.join(self.dir, 'positive') # doesnt work for multiple directories
            # self.neg_dir = os.path.join(self.dir, 'background') # doesnt work for multiple directories


            self.pos_images = [os.path.join(d, f) for d in self.pos_dirs for f in os.listdir(d)] # collects all the positive images
            self.neg_images = [os.path.join(d, f) for d in self.neg_dirs for f in os.listdir(d)] # collects all the negative images

            # self.pos_images = [os.path.join(self.pos_dir, f) for f in os.listdir(self.pos_dir)] # doesnt work for multiple directories
            # self.neg_images = [os.path.join(self.neg_dir, f) for f in os.listdir(self.neg_dir)] # doesnt work for multiple directories 
                  
            self.images = self.pos_images + self.neg_images
            self.labels = [1] * len(self.pos_images) + [0] * len(self.neg_images)
        elif mode == 'val':
            pass
        else:
            # self.dir = os.path.join(dir, 'test')
            # self.images = [os.path.join(self.dir, f) for f in os.listdir(self.dir)]
            pass # Pass for now as Proposals are only on training data

        # If no transform is specified, apply a default transform to convert to tensor
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor()           # Convert to tensor
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label