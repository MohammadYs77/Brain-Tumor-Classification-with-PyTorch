import sys
sys.path.append('D:/AI Courses/Semester 4/Brain Tumor/')
from configs import RESIZE, BATCH_SIZE

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, df, transform=None, shuffle_data=True):

        self.df = df
        self.transform = transform
        if shuffle_data:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple:
        img = Image.open(self.df.iloc[idx, 0]).convert("RGB")
        lbl = self.df.iloc[idx, 1]

        if self.transform:
            img = transforms.functional.resize(img, size=[RESIZE, RESIZE], interpolation=transforms.InterpolationMode.BICUBIC)
            img = self.transform(img)

        return img, lbl
    
    
def build_dataloader(tr_df, ts_df, data_transforms):

    tr_dt = ImageDataset(tr_df, data_transforms['train'])
    ts_dt = ImageDataset(ts_df, data_transforms['test'])

    tr_loader = DataLoader(tr_dt, batch_size=BATCH_SIZE, shuffle=True)
    ts_loader = DataLoader(ts_dt, batch_size=BATCH_SIZE, shuffle=True)

    dataloaders = {
        'train': tr_loader,
        'test': ts_loader
    }

    del tr_loader, ts_loader, tr_dt, ts_dt
    
    return dataloaders