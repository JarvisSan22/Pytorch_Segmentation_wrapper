from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os 
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch 
from PIL import Image
class Dataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            image_size=256,
            preprocessing=None,
            DEVICE="cuda"

    ):

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.images_fps = natsorted(os.listdir(images_dir))
        self.masks_fps  = natsorted(os.listdir(masks_dir))

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.DEVICE = DEVICE

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_dir +"/"+ self.images_fps[i])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.uint8(image))
        image = np.asarray(image.resize((self.image_size, self.image_size)))

        mask = cv2.imread(self.masks_dir +"/"+  self.masks_fps[i],0) #[x,y ]

        mask = np.uint8(cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST))


        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            #print(image.shape , image.type, mask.shape,mask.type)
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']


        image = torch.from_numpy(image).to(self.DEVICE)
        mask = torch.from_numpy(mask).to(self.DEVICE)

        return image, mask.long().unsqueeze(0)

    def __len__(self):
        return len(os.listdir(self.images_dir))