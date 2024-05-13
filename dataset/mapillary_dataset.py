import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms
import torch

class MapDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror

        self.id_to_trainid = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
                              16: 1, 17: 1, 18: 1, 19: 5, 20: 5, 21: 5, 22: 5, 23: 0, 24: 0, 27: 4, 29: 0, 30: 3,
                              43: 2, 44: 2, 45: 2, 46: 2, 47: 2, 48: 2, 49: 2, 50: 2,
                              52: 6, 53: 6, 54: 6, 55: 6, 56: 6, 57: 6, 58: 6, 59: 6, 60: 6, 61: 6, 62: 6}
                              
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            n_repeat = int(max_iters / len(self.img_ids))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
        self.files = []
        self.set = set
        
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            full_name = name.replace("images","labels")
            label_file = osp.join(self.root, full_name.split('.')[0]+'.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name.split('/')[-1]
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        if self.set != 'val':
            label = label.resize(self.crop_size, Image.NEAREST)
        else:
            label = label.resize((1920,1080), Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        if self.set == 'train' or self.set == 'val':
            label_copy = 255 * np.ones(label.shape, dtype=np.float32)        
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            label_copy = label

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(),np.array(size), name


if __name__ == '__main__':
    
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    composed_transforms = transforms.Compose([       
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    tgt_loader = data.DataLoader(
                    MapDataSet('/iarai/home/yonghao.xu/Data/SegmentationData/Mapillary/', './mapillary_train.txt',                
                    crop_size=(1024,512),
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=1, shuffle=True, num_workers=0,
                    pin_memory=True)

    channels_sum,channel_squared_sum = 0,0
    num_batches = len(tgt_loader)
    for data,_,_,_ in tgt_loader:
        channels_sum += torch.mean(data,dim=[0,2,3])   
        channel_squared_sum += torch.mean(data**2,dim=[0,2,3])       

    mean = channels_sum/num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5
    print(mean,std) 
    # tensor([-31.6067, -33.7499, -49.5125]) tensor([47.6189, 48.3803, 47.5466])
