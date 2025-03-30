from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, df, resize, transform=None, shuffle_data=True):

        self.df = df
        self.transform = transform
        self.resize = resize
        
        if shuffle_data:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple:
        img = Image.open(self.df.iloc[idx, 0]).convert("RGB")
        lbl = self.df.iloc[idx, 1]

        if self.transform:
            img = transforms.functional.resize(img, size=[self.resize, self.resize], interpolation=transforms.InterpolationMode.BICUBIC)
            img = self.transform(img)

        return img, lbl
    
    
def build_dataloader(tr_df, ts_df, resize, batch, data_transforms):

    tr_dt = ImageDataset(tr_df, resize, data_transforms['train'])
    ts_dt = ImageDataset(ts_df, resize, data_transforms['test'])

    tr_loader = DataLoader(tr_dt, batch_size=batch, shuffle=True)
    ts_loader = DataLoader(ts_dt, batch_size=batch, shuffle=True)

    dataloaders = {
        'train': tr_loader,
        'test': ts_loader
    }

    del tr_loader, ts_loader, tr_dt, ts_dt
    
    return dataloaders