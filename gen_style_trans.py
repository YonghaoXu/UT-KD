import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import os
from utils.tools import *
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from dataset.mapillary_dataset import MapDataSet
from model.Networks import StyleTransferNet,CustomizedInstanceNorm
from tqdm import tqdm

import warnings 
warnings.filterwarnings("ignore") 

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable
    """

    im = im_as_var.data.numpy().copy().transpose(1, 2, 0)
    im += IMG_MEAN
    im = im[:, :, ::-1] 
    im[im > 255] = 255
    im[im < 0] = 0
    im = Image.fromarray(np.uint8(im))
    return im

def get_arguments():
    parser = argparse.ArgumentParser()    
    #dataset
    parser.add_argument("--data_ID", type=int, default=1,
                        help="target dataset ID. 0: Cityscapes 1: IDD 2: Mapillary 3: GTA")
    parser.add_argument("--input_size", type=str, default='1024,512',
                        help="width and height of input images.")        

    #network
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="base learning rate.")
    parser.add_argument("--learning_rate_D", type=float, default=5e-5,
                        help="discriminator learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--restore_from", type=str, default='./pretrained_model/GTA_Trans_epoch_20_batch_6241.pth',
                        help="pretrained style transfer network T.")
    parser.add_argument("--src_IN", type=str, default='./pretrained_model/GTA_CIN_src_epoch_20_batch_6241.pth',
                        help="pretrained GTA style.")
    parser.add_argument("--tgt1_IN", type=str, default='./pretrained_model/GTA_CIN_tgt1_epoch_20_batch_6241.pth',
                        help="pretrained Cityscapes style.")
    parser.add_argument("--tgt2_IN", type=str, default='./pretrained_model/GTA_CIN_tgt2_epoch_20_batch_6241.pth',
                        help="pretrained IDD style.")
    parser.add_argument("--tgt3_IN", type=str, default='./pretrained_model/GTA_CIN_tgt3_epoch_20_batch_6241.pth',
                        help="pretrained Mapillary style.")
    parser.add_argument("--weight_decay", type=float, default=0.00005,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--set", type=str, default='train',
                        help="set")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./StyleTrans/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

def main():
    
    args = get_arguments()
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    if args.data_ID == 0:
        snapshot_dir = args.snapshot_dir+'Cityscapes/'+args.set+'/'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/Cityscapes/'
        data_list_target = './dataset/cityscapes_labellist_'+args.set+'.txt'

        tgt_loader = data.DataLoader(
                        cityscapesDataSet(data_dir_target, data_list_target,                  
                        crop_size=input_size,
                        scale=False, mirror=False, mean=IMG_MEAN,
                        set=args.set),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    elif args.data_ID == 1:
        snapshot_dir = args.snapshot_dir+'IDD/'+args.set+'/'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/IDD/IDD_Segmentation/'
        data_list_target = './dataset/IDD_'+args.set+'.txt'
        
        tgt_loader = data.DataLoader(
                        IDDDataSet(data_dir_target, data_list_target,               
                        crop_size=input_size,
                        scale=False, mirror=False, mean=IMG_MEAN,
                        set=args.set),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    elif args.data_ID == 2:
        snapshot_dir = args.snapshot_dir+'Map/'+args.set+'/'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/MapillaryVistas/'
        data_list_target = './dataset/Mapillary_'+args.set+'.txt'
        
        tgt_loader = data.DataLoader(
                        MapDataSet(data_dir_target, data_list_target,               
                        crop_size=input_size,
                        scale=False, mirror=False, mean=IMG_MEAN,
                        set=args.set),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    elif args.data_ID == 3:
        snapshot_dir = args.snapshot_dir+'GTA/'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/GTA5/'
        data_list_target = './dataset/GTA5_imagelist_train.txt'
                
        tgt_loader = data.DataLoader(
                        GTA5DataSet(data_dir_target, data_list_target,               
                        crop_size=input_size,
                        scale=False, mirror=False, mean=IMG_MEAN),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)

    original_size_tgt = '2048,1024'
    w, h = map(int, original_size_tgt.split(','))
    original_size_tgt = (w, h)

    # load network
    trans_net = StyleTransferNet(input_channel=3,output_channel=3)
    
    saved_state_dict = torch.load(args.restore_from,map_location='cuda:0')

    new_params = trans_net.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        new_params[j] = saved_state_dict[i]
    trans_net.load_state_dict(new_params)

    for name, param in trans_net.named_parameters():
        param.requires_grad=False
    trans_net = trans_net.cuda()
    
    
    customizedin_gta = CustomizedInstanceNorm()    
    saved_state_dict = torch.load(args.src_IN,map_location='cuda:0')
    new_params = customizedin_gta.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        new_params[j] = saved_state_dict[i]
    customizedin_gta.load_state_dict(new_params)
    for name, param in customizedin_gta.named_parameters():
        param.requires_grad=False
    customizedin_gta = customizedin_gta.cuda()

    customizedin_tgt1 = CustomizedInstanceNorm()    
    saved_state_dict = torch.load(args.tgt1_IN,map_location='cuda:0')
    new_params = customizedin_tgt1.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        new_params[j] = saved_state_dict[i]
    customizedin_tgt1.load_state_dict(new_params)
    for name, param in customizedin_tgt1.named_parameters():
        param.requires_grad=False
    customizedin_tgt1 = customizedin_tgt1.cuda()

    customizedin_tgt2 = CustomizedInstanceNorm()    
    saved_state_dict = torch.load(args.tgt2_IN,map_location='cuda:0')
    new_params = customizedin_tgt2.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        new_params[j] = saved_state_dict[i]
    customizedin_tgt2.load_state_dict(new_params)
    for name, param in customizedin_tgt2.named_parameters():
        param.requires_grad=False
    customizedin_tgt2 = customizedin_tgt2.cuda()
    
    customizedin_tgt3 = CustomizedInstanceNorm()    
    saved_state_dict = torch.load(args.tgt3_IN,map_location='cuda:0')
    new_params = customizedin_tgt3.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        new_params[j] = saved_state_dict[i]
    customizedin_tgt3.load_state_dict(new_params)
    for name, param in customizedin_tgt3.named_parameters():
        param.requires_grad=False
    customizedin_tgt3 = customizedin_tgt3.cuda()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    
    tbar = tqdm(tgt_loader)
    if args.data_ID == 0:
        for batch_index, tgt_data in enumerate(tbar):                   
            tbar.set_description('Trans: %d/%d image' % (batch_index+1, len(tgt_loader)))
            # target data loading
            tgt_img, _, _, tgt_name = tgt_data
            tgt_img = tgt_img.cuda()

            tgt1_2src = trans_net(tgt_img,customizedin_gta)    
            tgt1_2tgt2 = trans_net(tgt_img,customizedin_tgt2)    
            tgt1_2tgt3 = trans_net(tgt_img,customizedin_tgt3)        
            vis_tgt1_2src = recreate_image(tgt1_2src[0].cpu())    
            vis_tgt1_2tgt2 = recreate_image(tgt1_2tgt2[0].cpu())    
            vis_tgt1_2tgt3 = recreate_image(tgt1_2tgt3[0].cpu())  

            name_prefix = tgt_name[0].split('.')[0]
            name_suffix = tgt_name[0].split('.')[-1]

            im = vis_tgt1_2src.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2GTA.'+name_suffix)  
            im = vis_tgt1_2tgt2.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2IDD.'+name_suffix)  
            im = vis_tgt1_2tgt3.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2Map.'+name_suffix)  

    elif args.data_ID == 1:
        for batch_index, tgt_data in enumerate(tbar):                   
            tbar.set_description('Trans: %d/%d image' % (batch_index+1, len(tgt_loader)))
            # target data loading
            tgt_img, _, _, tgt_name = tgt_data
            tgt_img = tgt_img.cuda()

            tgt2_2src = trans_net(tgt_img,customizedin_gta)    
            tgt2_2tgt1 = trans_net(tgt_img,customizedin_tgt1)    
            tgt2_2tgt3 = trans_net(tgt_img,customizedin_tgt3)        
            vis_tgt2_2src = recreate_image(tgt2_2src[0].cpu())    
            vis_tgt2_2tgt1 = recreate_image(tgt2_2tgt1[0].cpu())    
            vis_tgt2_2tgt3 = recreate_image(tgt2_2tgt3[0].cpu())  

            name_prefix = tgt_name[0].split('.')[0]
            name_suffix = tgt_name[0].split('.')[-1]

            im = vis_tgt2_2src.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2GTA.'+name_suffix)  
            im = vis_tgt2_2tgt1.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2City.'+name_suffix)  
            im = vis_tgt2_2tgt3.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2Map.'+name_suffix)  
    
    elif args.data_ID == 2:
        for batch_index, tgt_data in enumerate(tbar):                   
            tbar.set_description('Trans: %d/%d image' % (batch_index+1, len(tgt_loader)))
            # target data loading
            tgt_img, _, _, tgt_name = tgt_data
            tgt_img = tgt_img.cuda()

            tgt3_2src = trans_net(tgt_img,customizedin_gta)    
            tgt3_2tgt1 = trans_net(tgt_img,customizedin_tgt1)    
            tgt3_2tgt2 = trans_net(tgt_img,customizedin_tgt2)        
            vis_tgt3_2src = recreate_image(tgt3_2src[0].cpu())    
            vis_tgt3_2tgt1 = recreate_image(tgt3_2tgt1[0].cpu())    
            vis_tgt3_2tgt2 = recreate_image(tgt3_2tgt2[0].cpu())  

            name_prefix = tgt_name[0].split('.')[0]
            name_suffix = tgt_name[0].split('.')[-1]

            im = vis_tgt3_2src.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2GTA.'+name_suffix)  
            im = vis_tgt3_2tgt1.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2City.'+name_suffix)  
            im = vis_tgt3_2tgt2.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2IDD.'+name_suffix)  
    
    elif args.data_ID == 3:
        for batch_index, tgt_data in enumerate(tbar):                   
            tbar.set_description('Trans: %d/%d image' % (batch_index+1, len(tgt_loader)))
            # target data loading
            tgt_img, _, _, tgt_name = tgt_data
            tgt_img = tgt_img.cuda()
 
            src_2tgt1 = trans_net(tgt_img,customizedin_tgt1)   
            src_2tgt2 = trans_net(tgt_img,customizedin_tgt2)       
            src_2tgt3 = trans_net(tgt_img,customizedin_tgt3)        
            vis_src_2tgt1 = recreate_image(src_2tgt1[0].cpu())    
            vis_src_2tgt2 = recreate_image(src_2tgt2[0].cpu())    
            vis_src_2tgt3 = recreate_image(src_2tgt3[0].cpu())  

            name_prefix = tgt_name[0].split('.')[0]
            name_suffix = tgt_name[0].split('.')[-1]

            im = vis_src_2tgt1.resize(original_size_tgt, Image.BICUBIC)        
            im.save(snapshot_dir+name_prefix+'_2City.'+name_suffix)  
            im = vis_src_2tgt2.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2IDD.'+name_suffix)  
            im = vis_src_2tgt3.resize(original_size_tgt, Image.BICUBIC)       
            im.save(snapshot_dir+name_prefix+'_2Map.'+name_suffix)  
    
if __name__ == '__main__':
    main()
