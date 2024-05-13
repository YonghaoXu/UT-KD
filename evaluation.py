import os
import argparse
import numpy as np
import torch
from torch.utils import data
from model.Networks import DeeplabMulti
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from dataset.mapillary_dataset import MapDataSet
from utils.tools import *
from tqdm import tqdm
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ID", type=int, default=1)
    parser.add_argument("--data_ID", type=int, default=1,
                        help="target dataset ID. 1: Cityscapes 2: IID 3: Mapillary")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="the index of the label to ignore in the training.")
    parser.add_argument("--input_size", type=str, default='1024,512',
                        help="width and height of input images.")      
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--num_classes", type=int, default=7,
                        help="number of classes.")
    parser.add_argument("--restore_from", type=str, default='./pretrained_model/MT-KD/synthetic2real_GTA2CityIDDMapillary/Seg_batch_11000_miou_7633_val2_miou_7084_val3_miou_7541.pth',
                        help="restored model.")
    parser.add_argument("--snapshot_dir", type=str, default='./Maps/',
                        help="Path to save result.")
    return parser.parse_args()

def main():
    args = get_arguments()
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    if args.data_ID == 1:
        data_list_target = './dataset/cityscapes_labellist_val.txt'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/Cityscapes/'     
        input_size_target = (2048,1024)
             
        testloader = data.DataLoader(
                        cityscapesDataSet(data_dir_target, data_list_target,                  
                        crop_size=input_size,
                        scale=False, mirror=False, mean=IMG_MEAN,
                        set='val',num_classes=args.num_classes),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)
        
        snapshot_dir = args.snapshot_dir+'Cityscapes/'

    elif args.data_ID == 2:
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/IDD/IDD_Segmentation/'
        data_list_target = './dataset/IDD_val.txt'
        input_size_target = (1920,1080)
         
        testloader = data.DataLoader(
                        IDDDataSet(data_dir_target, data_list_target,               
                        crop_size=input_size,
                        scale=False, mirror=False, mean=IMG_MEAN,
                        set='val'),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

        snapshot_dir = args.snapshot_dir+'IDD/' 

    elif args.data_ID == 3:
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/MapillaryVistas/'
        data_list_target = './dataset/Mapillary_val.txt'
        input_size_target = (1920,1080)
          
        testloader = data.DataLoader(
                        MapDataSet(data_dir_target, data_list_target,               
                        crop_size=input_size,
                        scale=False, mirror=False, mean=IMG_MEAN,
                        set='val'),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

        snapshot_dir = args.snapshot_dir+'Mapillary/'  

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'Evaluation.txt', 'w')

    model = DeeplabMulti(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    test_mIoU7(f,model, testloader, 0,input_size_target,print_per_batches=100)
    
    #Save the segmentation maps
    interp = nn.Upsample(size=(1024,2048), mode='bilinear')
    tbar = tqdm(testloader)
    for index, batch in enumerate(tbar):                   
        tbar.set_description('Test: %d/%d image' % (index+1, len(testloader)))
        image, _,_, name = batch
        _,output = model(image.cuda())
        output = interp(output).cpu().data[0].numpy()   
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        output_col = colorize_mask_7(output)
        name = name[0].split('/')[-1]
        output_col.save('%s/%s_%s.png' % (snapshot_dir, name.split('.')[0],'predict'))

    f.close()

if __name__ == '__main__':
    main()
