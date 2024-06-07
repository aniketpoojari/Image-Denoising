import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


# DEFINE CUSTOM CLASS
class ImgDataset(Dataset):
    def __init__(self, img_dir):
        self.dir = img_dir

        self.imgs_name = []
        for img_name in os.listdir(self.dir + "clean"):
            self.imgs_name.append(img_name)

    def __getitem__(self, i):

        clean = Image.open(self.dir + "clean/" + self.imgs_name[i])
        noisy = Image.open(self.dir + "noisy/" + self.imgs_name[i])

        clean = ToTensor()(clean)
        noisy = ToTensor()(noisy)

        return noisy, clean

    def __len__(self):
        return len(self.imgs_name)


def get_data_loader(image_dir, batch_size):

    dataset = ImgDataset(image_dir)

    loader = DataLoader(dataset, batch_size=batch_size)

    return loader
