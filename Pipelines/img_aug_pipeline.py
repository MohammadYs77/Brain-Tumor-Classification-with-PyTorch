from torchvision import transforms


def make_augmentation_pipeline(resize):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': 
            transforms.Compose([
                transforms.CenterCrop(resize),
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
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                normalize
            ])
    }
    
    return data_transforms
