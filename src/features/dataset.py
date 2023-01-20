from glob import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FlickrDataset(Dataset):
    def __init__(self, root: str, transform):
        self.filenames = glob(f"{root}/*.jpg")
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        return self.transform(image)

    def __len__(self):
        return len(self.filenames)


content_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=(0.6, 1.3), contrast=(0.7, 1.4)),
    transforms.RandomRotation((-10, 10)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255)),
])

style_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255)),
])
