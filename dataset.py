"""
数据集 dataset
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import os


def load_image_list(image_dir: str) -> list:
    return_list = next(os.walk(image_dir), (None, None, []))[2]
    return return_list


class SunData(Dataset):
    def __init__(self, root, imList):
        super(SunData, self).__init__()
        self.root = root
        self.imList = imList
        self.preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_img(self, path):
        img_pil = utils.Image.open(path).convert('L')
        img_pil = img_pil.crop((78, 76, 1078, 1076))
        # img_pil = img_pil.resize((500, 500))
        img_tensor = self.preprocess(img_pil)
        return img_tensor

    def __getitem__(self, item):
        img_path = self.root + "/" + self.imList[item]
        img = self.load_img(img_path)
        return img

    def __len__(self):
        return len(self.imList)


if __name__ == '__main__':
    test = SunData("data/train", load_image_list("data/train"))
    print(test.__getitem__(5).shape)
    trainLoader = DataLoader(test)
