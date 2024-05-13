import os.path as osp
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms

class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None,crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, num_classes=7):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(max_iters / len(self.img_ids))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []
        if num_classes==19:
            self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                                19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                                26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        elif num_classes==7:
            self.id_to_trainid = {7: 0, 8: 0, 11: 1, 12: 1, 13: 1, 17: 2,
                                19: 2, 20: 2, 21: 3, 22: 3, 23: 4, 24: 5, 25: 5,
                                26: 6, 27: 6, 28: 6, 31: 6, 32: 6, 33: 6}

        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name

          
if __name__ == '__main__':
    
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    composed_transforms = transforms.Compose([       
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    src_loader = data.DataLoader(
                    GTA5DataSet('/iarai/home/yonghao.xu/Data/SegmentationData/GTA5/', './GTA5_imagelist_train.txt',
                    crop_size=(1024,512),
                    scale=False, mirror=False, mean=IMG_MEAN, transform=composed_transforms),
                    batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    channels_sum,channel_squared_sum = 0,0
    num_batches = len(src_loader)
    for data,_,_,_ in src_loader:
        channels_sum += torch.mean(data,dim=[0,2,3])   
        channel_squared_sum += torch.mean(data**2,dim=[0,2,3])       

    mean = channels_sum/num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5
    print(mean,std) 
    #tensor([-3.4262, -3.4262, -3.4262]) tensor([63.8150, 63.8150, 63.8150])
