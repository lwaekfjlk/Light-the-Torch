import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

def pick(probability, base=0):
    compare = random.random()
    return True if probability + base > compare else False

def random_split_percent(dataset, percentages):
    if round(sum(percentages), 3) != 1.0:
        raise ValueError('Sum of input percentages({}) does not equal 1'.format(sum(percentages)))
    lenList = list(map(lambda x : int(x * len(dataset)), percentages))
    lenList[-1] = len(dataset) - sum(lenList[0:-1])
    return random_split(dataset, lenList)

class ImageSet(Dataset):
    def __init__(self, raw_root, label_root, width=480, height=320):
        self.raw_root, self.label_root = raw_root, label_root
        self.height, self.width = height, width
        self.namelist = os.listdir(raw_root)

    def __getitem__(self, i):
        name = self.namelist[i]
        fname, ext = os.path.splitext(name)
        try:
            imgIn = Image.open(os.path.join(self.raw_root, name))
            imgOut = Image.open(os.path.join(self.label_root, fname + "_bin.png"))
        except FileNotFoundError:
            print("the name of raw and label image don't correspond")
            print(self.namelist[i])
            return 
        assert imgIn.size == imgOut.size
        
        imgWidth, imgHeight = imgIn.size
        while True:
            left = random.randint(0, imgWidth - self.width)
            upper = random.randint(0, imgHeight - self.height)
            imgCrop = np.vectorize(lambda x: 0 if (x == 0 or x == 255) else 1)(
                    np.array(
                        imgOut.crop([left, upper, left + self.width, upper + self.height])
                    )
                )
            if pick(np.average(imgCrop) * 10, 0.1):
                plt.imshow(imgCrop)
                imgOut = torch.from_numpy(imgCrop)
                break
        
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        imgIn = trans(imgIn.crop([left, upper, left + self.width, upper + self.height]))

        return imgIn, imgOut

    def __len__(self):
        return len(self.namelist)

if __name__ == '__main__':
    pass