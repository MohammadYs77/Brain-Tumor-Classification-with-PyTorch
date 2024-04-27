import sys
sys.path.append('D:/AI Courses/Semester 4/Brain Tumor/')
from configs import RESIZE

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data_transforms = {
    'train': 
        transforms.Compose([
            transforms.CenterCrop(RESIZE),
            #transforms.Resize((RESIZE, RESIZE)),
#              transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#              transforms.RandomHorizontalFlip(),
#              transforms.RandomRotation(20),
#              transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            normalize
        ]),
    'test': 
        transforms.Compose([
            transforms.CenterCrop(RESIZE),
            transforms.ToTensor(),
            normalize
        ])
}
