from torch.utils.data import Dataset
import os
from PIL import Image

class CroppedProposalDataset(Dataset):
    def __init__(self, dir, mode, transform=None):
        if mode == 'train':
            self.dir = os.path.join(dir, 'train')
            self.pos_dir = os.path.join(self.dir, 'positive')
            self.neg_dir = os.path.join(self.dir, 'background')
            self.pos_images = [os.path.join(self.pos_dir, f) for f in os.listdir(self.pos_dir)]
            self.neg_images = [os.path.join(self.neg_dir, f) for f in os.listdir(self.neg_dir)]        
            self.images = self.pos_images + self.neg_images    
        elif mode == 'val':
            pass
        else:
            self.dir = os.path.join(dir, 'test')
            self.images = [os.path.join(self.dir, f) for f in os.listdir(self.dir)]

        self.transform = transform
    # TODO: ADD LABELS and complete dataloader
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label