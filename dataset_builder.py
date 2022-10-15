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
        j=1
        while len(contours)<= 60 :
            #if an image is not recognized then an all black image will be created and it's label will be set as that of the joker (we are not looking to sort jokers)
            contours = utils.find_contours(image_path, thresh= 170-j*20)
            j += 1

        image = utils.isolate_resize_card(image_path, contours, (172, 172))

        pre_label = json_dictionary["images"][idx]["file_name"]
        pre_label = utils.labels_from_text(pre_label) 
        label = np.zeros(53) 
        label[pre_label-1] = 1      
        file.close()
        return image, label



##!!!!!!!!!!!!!!!!!!!!!!!!!! did not normalize data yet #TODO


if __name__ == "__main__":
    for i in range(100):
        image, label = C.__getitem__(i)


    