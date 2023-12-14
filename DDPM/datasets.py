import cv2
import torch.utils.data
from torchvision import transforms
from os import listdir
from os.path import join
import numpy as np


class CelebAData(torch.utils.data.Dataset):
    def __init__(self, dir, image_size):
        super().__init__()
        self.dir = dir
        self.list = listdir(self.dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: (x*2 - 1))
        ])

    def __getitem__(self, item):
        image_path = join(self.dir, self.list[item])
        b, g, r = cv2.split(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
        image = self.transform(cv2.merge([r, g, b]))
        return image

    def __len__(self):
        return len(self.list)


class MNISTData(torch.utils.data.Dataset):
    def __init__(self,
                 file_npy,
                 image_size=32):
        super(MNISTData, self).__init__()
        # load data
        self.image_size = image_size
        self.dataset = np.load(file_npy)

    def __getitem__(self, item):
        image = cv2.resize(self.dataset[item, :, :], (self.image_size, self.image_size)).astype(np.float32)
        return torch.from_numpy(image / 255 * 2 - 1).unsqueeze(dim=0)

    def __len__(self):
        return self.dataset.shape[0]


if __name__ == '__main__':
    dir = "D:/neural network/Dataset/celebA/Img/img_align_celeba/"
    image_size = 128
    dataset = CelebAData(dir, image_size)
    image = dataset[0].numpy().transpose(1, 2, 0)
    cv2.imshow('image', image)
    cv2.waitKey()