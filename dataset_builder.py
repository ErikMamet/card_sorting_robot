import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import json
import utils
import os.path as osp
import numpy as np

class Cards(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        file = open(self.img_labels)
        json_dictionary = json.load(file)
        image_path = osp.join(self.img_dir,json_dictionary["images"][idx]["file_name"])
        contours = utils.find_contours(image_path, thresh= 170)
        temp = 1
        while contours == 0 :
            contours = utils.find_contours(image_path, thresh= 170-30*temp)
            temp += 1

        image = utils.isolate_resize_card(image_path, contours, (172, 172))

        pre_label = json_dictionary["images"][idx]["file_name"]
        pre_label = utils.labels_from_text(pre_label) 
        label = np.zeros(53) 
        label[pre_label-1] = 1      
        file.close()
        return image, label


########
### split test and train
########

batch_size = 64
validation_split_index = .2
seed= 42
np.random.seed(seed)

# Dataset
C = Cards("./annotation.json", "./dataset/Images/Images")

# Creating data indices for training and validation split_indexs:
dataset_size = C.__len__()
indices = list(range(dataset_size))
split_index = int(np.floor(validation_split_index * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split_index:], indices[:split_index]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(C, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(C, batch_size=batch_size, sampler=valid_sampler)


##!!!!!!!!!!!!!!!!!!!!!!!!!! did not normalize data yet #TODO


if __name__ == "__main__":
    for i in range(100):
        image, label = C.__getitem__(i)


    