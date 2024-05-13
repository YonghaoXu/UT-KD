import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from utils.tools import *
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from dataset.mapillary_dataset import MapDataSet
from model.Networks import DeeplabMulti,Discriminator
import torch.backends.cudnn as cudnn
import mask_gen

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():
    parser = argparse.ArgumentParser()
    #dataset
    parser.add_argument("--data_ID", type=int, default=1)
    parser.add_argument("--input_size_tgt", type=str, default='1024,512',
                        help="width and height of input target images.")                       
    parser.add_argument("--num_classes", type=int, default=7,
                        help="number of classes.")

    #network
    parser.add_argument("--batch_size", type=int, default=2,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5,
                        help="base learning rate.")
    parser.add_argument("--learning_rate_D", type=float, default=1e-5,
                        help="discriminator learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_steps", type=int, default=50000,
                        help="Number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=50000,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--restore_from", type=str, default='./pretrained_model/GTA_pretrain_batch_50000_miou_67.pth',
                        help="pretrained model from GTA-5.")
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="regularisation parameter for L2-loss.")

    #hyperparameters
    parser.add_argument("--teacher_alpha", type=float, default=0.999,
                        help="teacher alpha in EMA.")
    parser.add_argument("--con_weight", type=float, default=100,
                        help="consistency regularization weight.")
    parser.add_argument("--out_weight", type=float, default=0.1,
                        help="adversarial training weight.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./UT-KD/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

def main():

    args = get_arguments()
    w, h = map(int, args.input_size_tgt.split(','))
    input_size_tgt = (w, h)

    if args.data_ID == 1:
        # synthetic2real_GTA2City2IDD
        seg_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2City/Seg_batch_12000_miou_7631.pth'
        D1_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2City/D_net1_batch_12000.pth'
        D2_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2City/D_net2_batch_12000.pth'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/IDD/IDD_Segmentation/'
        data_list_val = './dataset/IDD_val.txt'
        data_list_target = './dataset/IDD_train.txt'
        input_size_test = (1920,1080)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2City2IDD/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    IDDDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    IDDDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    

    elif args.data_ID == 2:
        # synthetic2real_GTA2City2Mapillary
        seg_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2City/Seg_batch_12000_miou_7631.pth'
        D1_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2City/D_net1_batch_12000.pth'
        D2_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2City/D_net2_batch_12000.pth'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/MapillaryVistas/'
        data_list_val = './dataset/Mapillary_val.txt'
        data_list_target = './dataset/Mapillary_train.txt'
        input_size_test = (1920,1080)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2City2Mapillary/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    MapDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    MapDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    

    elif args.data_ID == 3:
        # synthetic2real_GTA2IDD2City
        seg_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2IDD/Seg_batch_20500_miou_7145.pth'
        D1_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2IDD/D_net1_batch_20500.pth'
        D2_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2IDD/D_net2_batch_20500.pth'
        data_list_target = './dataset/cityscapes_labellist_train.txt'
        data_list_val = './dataset/cityscapes_labellist_val.txt'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/Cityscapes/'     
        input_size_test = (2048,1024)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2IDD2City/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    cityscapesDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train',num_classes=args.num_classes),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    cityscapesDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val',num_classes=args.num_classes),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    

    elif args.data_ID == 4:
        # synthetic2real_GTA2Mapillary2City
        seg_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2Mapillary/Seg_batch_15500_miou_7523.pth'
        D1_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2Mapillary/D_net1_batch_15500.pth'
        D2_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2Mapillary/D_net2_batch_15500.pth'
        data_list_target = './dataset/cityscapes_labellist_train.txt'
        data_list_val = './dataset/cityscapes_labellist_val.txt'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/Cityscapes/'     
        input_size_test = (2048,1024)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2Mapillary2City/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    cityscapesDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train',num_classes=args.num_classes),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    cityscapesDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val',num_classes=args.num_classes),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    

    elif args.data_ID == 5:
        # synthetic2real_GTA2IDDMapillary2City
        seg_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2IDDMapillary/Seg_batch_15500_miou_7090_val2_miou_7531.pth'
        D1_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2IDDMapillary/D_net1_batch_15500.pth'
        D2_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2IDDMapillary/D_net2_batch_15500.pth'
        data_list_target = './dataset/cityscapes_labellist_train.txt'
        data_list_val = './dataset/cityscapes_labellist_val.txt'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/Cityscapes/'     
        input_size_test = (2048,1024)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2IDDMapillary2City/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    cityscapesDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train',num_classes=args.num_classes),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    cityscapesDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val',num_classes=args.num_classes),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    

    elif args.data_ID == 6:
        # synthetic2real_GTA2CityMapillary2IDD
        seg_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2CityMapillary/Seg_batch_9000_miou_7683_val2_miou_7536.pth'
        D1_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2CityMapillary/D_net1_batch_9000.pth'
        D2_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2CityMapillary/D_net2_batch_9000.pth'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/IDD/IDD_Segmentation/'
        data_list_val = './dataset/IDD_val.txt'
        data_list_target = './dataset/IDD_train.txt'
        input_size_test = (1920,1080)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2CityMapillary2IDD/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    IDDDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    IDDDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    

    elif args.data_ID == 7:
        # synthetic2real_GTA2CityIDD2Mapillary
        seg_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2CityIDD/Seg_batch_14000_miou_7654_val2_miou_7122.pth'
        D1_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2CityIDD/D_net1_batch_14000.pth'
        D2_restore_from = './pretrained_model/MT-KD/synthetic2real_GTA2CityIDD/D_net2_batch_14000.pth'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/MapillaryVistas/'
        data_list_val = './dataset/Mapillary_val.txt'
        data_list_target = './dataset/Mapillary_train.txt'
        input_size_test = (1920,1080)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2CityIDD2Mapillary/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    MapDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    MapDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    

    elif args.data_ID == 8:
        # real2real_City2Mapillary2IDD
        seg_restore_from = './pretrained_model/MT-KD/real2real_City2Mapillary/Seg_batch_12000_miou_7527.pth'
        D1_restore_from = './pretrained_model/MT-KD/real2real_City2Mapillary/D_net1_batch_12000.pth'
        D2_restore_from = './pretrained_model/MT-KD/real2real_City2Mapillary/D_net2_batch_12000.pth'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/IDD/IDD_Segmentation/'
        data_list_val = './dataset/IDD_val.txt'
        data_list_target = './dataset/IDD_train.txt'
        input_size_test = (1920,1080)

        snapshot_dir = args.snapshot_dir+'real2real_City2Mapillary2IDD/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    IDDDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    IDDDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    

    elif args.data_ID == 9:
        # real2real_City2IDD2Mapillary
        seg_restore_from = './pretrained_model/MT-KD/real2real_City2IDD/Seg_batch_33500_miou_7245.pth'
        D1_restore_from = './pretrained_model/MT-KD/real2real_City2IDD/D_net1_batch_33500.pth'
        D2_restore_from = './pretrained_model/MT-KD/real2real_City2IDD/D_net2_batch_33500.pth'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/MapillaryVistas/'
        data_list_val = './dataset/Mapillary_val.txt'
        data_list_target = './dataset/Mapillary_train.txt'
        input_size_test = (1920,1080)

        snapshot_dir = args.snapshot_dir+'real2real_City2IDD2Mapillary/UTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
        tgt_loader = data.DataLoader(
                    MapDataSet(data_dir_target, data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

        val_loader = data.DataLoader(
                    MapDataSet(data_dir_target, data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)    
        
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'log.txt', 'w')

    # Create network
    
    student_net = DeeplabMulti(num_classes=args.num_classes)
    teacher_net = DeeplabMulti(num_classes=args.num_classes)
    D_net1 = Discriminator(input_channel=args.num_classes)
    D_net2 = Discriminator(input_channel=args.num_classes)
        
    saved_state_dict = torch.load(D1_restore_from)       
    D_net1.load_state_dict(saved_state_dict)
    saved_state_dict = torch.load(D2_restore_from)       
    D_net2.load_state_dict(saved_state_dict)   
    D_net1.eval()
    D_net2.eval()
    for name, param in D_net1.named_parameters():    
        param.requires_grad=False
    for name, param in D_net2.named_parameters():    
        param.requires_grad=False

    saved_state_dict = torch.load(seg_restore_from)     # pre-trained MT-KD model
    teacher_net.load_state_dict(saved_state_dict)
    saved_state_dict = torch.load(args.restore_from)    # from scratch
    student_net.load_state_dict(saved_state_dict)    

    for name, param in teacher_net.named_parameters():    
        param.requires_grad=False

    student_net = student_net.cuda()
    teacher_net = teacher_net.cuda()
    D_net1 = D_net1.cuda()
    D_net2 = D_net2.cuda()
   
    optimizer = optim.SGD(student_net.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    
    student_params = list(student_net.parameters())
    teacher_params = list(teacher_net.parameters())

    teacher_optimizer = WeightEMA(
        teacher_params, 
        student_params,
        alpha=args.teacher_alpha,
    )

    interp_tgt = nn.Upsample(size=(input_size_tgt[1], input_size_tgt[0]),  mode='bilinear', align_corners=True)
    n_class = args.num_classes
    loss_hist = np.zeros((args.num_steps_stop,7))

    mIoU_hist = 20
    con_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    cudnn.enabled = True
    cudnn.benchmark = True
    student_net.eval()
    teacher_net.eval()
    half_batch = args.batch_size // 2
    mask_generator = mask_gen.BoxMaskGenerator(prop_range=0.5, n_boxes=1,
                                            random_aspect_ratio=not False,
                                            prop_by_area=not False, within_bounds=not False,
                                            invert=not False)
                        
    for batch_index, tgt_data in enumerate(tgt_loader):
        if batch_index>=args.num_steps_stop:
            break
        
        decay_adv = (1 - batch_index/args.num_steps)
        tem_time = time.time()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,args.learning_rate,batch_index,args.num_steps)
                
        # L_con
        images, _,_, _ = tgt_data
        images = images.cuda()        
        
        images_1 = images[:half_batch,:,:,:]
        images_2 = images[half_batch:,:,:,:]
        
        mask_size = images.shape[2:]
        # (N,1,H,W)
        batch_mix_masks = torch.from_numpy(mask_generator.generate_params(len(images_1), mask_size)).cuda().float()
            
        # Mix images with masks
        mixed_s = images_1 * (1 - batch_mix_masks) + images_2 * batch_mix_masks
       
        _,tgt_s_output_mixed_s = student_net(mixed_s)
        _,tgt_t_output_ori1 = teacher_net(images_1)
        _,tgt_t_output_ori2 = teacher_net(images_2)

        tgt_t_output = interp_tgt(tgt_t_output_ori1) * (1 - batch_mix_masks) + interp_tgt(tgt_t_output_ori2) * batch_mix_masks
        tgt_s_output = interp_tgt(tgt_s_output_mixed_s)

        tgt_t_predicts = F.softmax(tgt_t_output, dim=1).transpose(1, 2).transpose(2, 3)
        tgt_s_predicts = F.softmax(tgt_s_output, dim=1).transpose(1, 2).transpose(2, 3)

        tgt_s_predicts = tgt_s_predicts.contiguous().view(-1,n_class)
        tgt_t_predicts = tgt_t_predicts.contiguous().view(-1,n_class)
        
        con_loss_value = con_loss(tgt_s_predicts, tgt_t_predicts)

        tgt_s_output_ori1,tgt_s_output_ori2 = student_net(images)
        
        tgt_s_output1 = interp_tgt(tgt_s_output_ori1)
        tgt_s_output2 = interp_tgt(tgt_s_output_ori2)

        D_out1 = D_net1(F.softmax(tgt_s_output1, dim=1))
        D_out2 = D_net2(F.softmax(tgt_s_output2, dim=1))

        loss_adv_tgt1 = bce_loss(D_out1,
                                    Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

        loss_adv_tgt2 = bce_loss(D_out2,
                                    Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())

        loss_adv_G = (loss_adv_tgt1*0.1 + loss_adv_tgt2)
    
        # Weighted sum     
        total_loss = args.con_weight * con_loss_value + args.out_weight * loss_adv_G * decay_adv            
    
        loss_hist[batch_index,0] = total_loss.item()
        loss_hist[batch_index,2] = con_loss_value.item()
        loss_hist[batch_index,3] = loss_adv_G.item()

        total_loss.backward()

        optimizer.step()
        teacher_optimizer.step()   

        batch_time = time.time()-tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d time: %.2f con_loss = %.3f g_loss = %.3f \n'\
                %(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,2]),\
                        np.mean(loss_hist[batch_index-9:batch_index+1,3])))
            f.write('Iter %d/%d time: %.2f con_loss = %.3f g_loss = %.3f \n'\
                %(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,2]),\
                        np.mean(loss_hist[batch_index-9:batch_index+1,3])))
            f.flush() 
            
        if (batch_index+1) % 500 == 0:                    
            mIoU_new = test_mIoU7(f,teacher_net, val_loader, batch_index+1,input_size_test,print_per_batches=10)
            
            # Saving the models        
            if mIoU_new > mIoU_hist:    
                f.write('Save Model\n') 
                print('Save Model')                     
                model_name = 'batch_'+repr(batch_index+1)+'_miou_'+repr(int(mIoU_new*100))+'.pth'
                torch.save(teacher_net.state_dict(), os.path.join(
                    snapshot_dir, model_name))      
                mIoU_hist = mIoU_new

    f.close()
    np.savez(snapshot_dir+'loss.npz',loss_hist=loss_hist) 
    
if __name__ == '__main__':
    main()
