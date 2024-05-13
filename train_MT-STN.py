import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import os
import time
from utils.tools import *
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from dataset.mapillary_dataset import MapDataSet
from model.Networks import StyleTransferNet,CustomizedInstanceNorm,Discriminator
from torch.utils.tensorboard import SummaryWriter 
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore") 
import torch.distributed as dist

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
    parser.add_argument("--data_dir_source", type=str, default='/iarai/home/yonghao.xu/Data/SegmentationData/GTA5/',
                        help="source dataset path.")
    parser.add_argument("--data_list_source", type=str, default='./dataset/GTA5_imagelist_train.txt',
                        help="source dataset list file.")
    parser.add_argument("--data_dir_target1", type=str, default='/iarai/home/yonghao.xu/Data/SegmentationData/Cityscapes/',
                        help="target1 dataset path.")
    parser.add_argument("--data_list_target1", type=str, default='./dataset/cityscapes_labellist_train.txt',
                        help="target1 dataset list file.")
    parser.add_argument("--data_dir_target2", type=str, default='/iarai/home/yonghao.xu/Data/SegmentationData/IDD/IDD_Segmentation/',
                        help="target2 dataset path.")
    parser.add_argument("--data_list_target2", type=str, default='./dataset/IDD_train.txt',
                        help="target2 dataset list file.")
    parser.add_argument("--data_dir_target3", type=str, default='/iarai/home/yonghao.xu/Data/SegmentationData/MapillaryVistas/',
                        help="target3 dataset path.")
    parser.add_argument("--data_list_target3", type=str, default='./dataset/Mapillary_train.txt',
                        help="target3 dataset list file.")
    parser.add_argument("--input_size", type=str, default='832,416',#'1024,512',
                        help="width and height of input images.")       
    parser.add_argument("--original_size_src", type=str, default='1914,1052',
                        help="width and height of original images.")                       
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes.")
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')

    #network
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="base learning rate.")
    parser.add_argument("--learning_rate_D", type=float, default=1e-5,
                        help="discriminator learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_epoch", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.00005,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--rec_weight", type=float, default=1000,
                        help="reconstruction weight.")
    #result
    parser.add_argument("--snapshot_dir", type=str, default='./MT-STN/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

def main():
    
    args = get_arguments()
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    snapshot_dir = args.snapshot_dir+'rec_weight_'+str(args.rec_weight)+'_Glr'+str(args.learning_rate)+'_Dlr'+str(args.learning_rate_D)+'time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.original_size_src.split(','))
    original_size_src = (w, h)
    original_size_tgt1 = (2048, 1024)
    original_size_tgt2 = (1920,1080)
    original_size_tgt3 = (3840,2160)

    src_dataset = GTA5DataSet(args.data_dir_source, args.data_list_source,
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN)
    src_sampler = torch.utils.data.distributed.DistributedSampler(src_dataset)
    tgt_dataset1 = cityscapesDataSet(args.data_dir_target1, args.data_list_target1, max_iters=24964,                  
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train')
    tgt_sampler1 = torch.utils.data.distributed.DistributedSampler(tgt_dataset1)
    tgt_dataset2 = IDDDataSet(args.data_dir_target2, args.data_list_target2, max_iters=24964,                  
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train')
    tgt_sampler2 = torch.utils.data.distributed.DistributedSampler(tgt_dataset2)
    tgt_dataset3 = MapDataSet(args.data_dir_target3, args.data_list_target3, max_iters=24964,                  
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train')
    tgt_sampler3 = torch.utils.data.distributed.DistributedSampler(tgt_dataset3)

    src_loader = data.DataLoader(src_dataset,batch_size=args.batch_size, \
                    shuffle=(src_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=src_sampler)

    tgt_loader1 = data.DataLoader(tgt_dataset1,batch_size=args.batch_size, \
                    shuffle=(tgt_sampler1 is None), num_workers=args.num_workers,pin_memory=True, sampler=tgt_sampler1)
    tgt_loader2 = data.DataLoader(tgt_dataset2,batch_size=args.batch_size, \
                    shuffle=(tgt_sampler2 is None), num_workers=args.num_workers,pin_memory=True, sampler=tgt_sampler2)
    tgt_loader3 = data.DataLoader(tgt_dataset3,batch_size=args.batch_size, \
                    shuffle=(tgt_sampler3 is None), num_workers=args.num_workers,pin_memory=True, sampler=tgt_sampler3)

    num_batches = len(src_loader)


    # Create network
    trans_net = StyleTransferNet(input_channel=3,output_channel=3).cuda()  
    customizedin_src = CustomizedInstanceNorm().cuda()  
    customizedin_tgt1 = CustomizedInstanceNorm().cuda()  
    customizedin_tgt2 = CustomizedInstanceNorm().cuda()  
    customizedin_tgt3 = CustomizedInstanceNorm().cuda()  
    D_net_src = Discriminator(input_channel=3).cuda()  
    D_net_tgt1 = Discriminator(input_channel=3).cuda()  
    D_net_tgt2 = Discriminator(input_channel=3).cuda()  
    D_net_tgt3 = Discriminator(input_channel=3).cuda()  

    trans_net = nn.parallel.DistributedDataParallel(trans_net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    customizedin_src = nn.parallel.DistributedDataParallel(customizedin_src, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    customizedin_tgt1 = nn.parallel.DistributedDataParallel(customizedin_tgt1, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    customizedin_tgt2 = nn.parallel.DistributedDataParallel(customizedin_tgt2, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    customizedin_tgt3 = nn.parallel.DistributedDataParallel(customizedin_tgt3, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    D_net_src = nn.parallel.DistributedDataParallel(D_net_src, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    D_net_tgt1 = nn.parallel.DistributedDataParallel(D_net_tgt1, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    D_net_tgt2 = nn.parallel.DistributedDataParallel(D_net_tgt2, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    D_net_tgt3 = nn.parallel.DistributedDataParallel(D_net_tgt3, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    optimizer_trans = optim.Adam(trans_net.parameters(),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_trans.zero_grad()
    
    optimizer_in_src = optim.Adam(customizedin_src.parameters(),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_in_src.zero_grad()
    optimizer_in_tgt1 = optim.Adam(customizedin_tgt1.parameters(),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_in_tgt1.zero_grad()
    optimizer_in_tgt2 = optim.Adam(customizedin_tgt2.parameters(),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_in_tgt2.zero_grad()
    optimizer_in_tgt3 = optim.Adam(customizedin_tgt3.parameters(),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_in_tgt3.zero_grad()

    optimizer_D_src = optim.Adam(D_net_src.parameters(), lr=args.learning_rate_D, weight_decay=args.weight_decay)
    optimizer_D_src.zero_grad()
    optimizer_D_tgt1 = optim.Adam(D_net_tgt1.parameters(), lr=args.learning_rate_D, weight_decay=args.weight_decay)
    optimizer_D_tgt1.zero_grad()
    optimizer_D_tgt2 = optim.Adam(D_net_tgt2.parameters(), lr=args.learning_rate_D, weight_decay=args.weight_decay)
    optimizer_D_tgt2.zero_grad()
    optimizer_D_tgt3 = optim.Adam(D_net_tgt3.parameters(), lr=args.learning_rate_D, weight_decay=args.weight_decay)
    optimizer_D_tgt3.zero_grad()

    num_steps = args.num_epoch*num_batches
    loss_hist = np.zeros((num_steps,7))

    recons_loss = torch.nn.L1Loss(reduction='mean')
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    fake_label = 0
    real_label = 1

    trans_net.train()
    D_net_src.train()
    customizedin_src.train()
    customizedin_tgt1.train()
    customizedin_tgt2.train()
    customizedin_tgt3.train()

    writer = SummaryWriter(log_dir=snapshot_dir)
    global_step = 0
    for epoch in range(args.num_epoch):
        for batch_index, (src_data, tgt1_data, tgt2_data, tgt3_data) in enumerate(zip(src_loader, tgt_loader1, tgt_loader2, tgt_loader3)):

            tem_time = time.time()
                   
            # source data loading
            src_img, _, _, src_name = src_data
            src_img = src_img.cuda()

            # target data loading
            tgt1_img, _, _, tgt1_name = tgt1_data
            tgt1_img = tgt1_img.cuda()
            tgt2_img, _, _, tgt2_name = tgt2_data
            tgt2_img = tgt2_img.cuda()
            tgt3_img, _, _, tgt3_name = tgt3_data
            tgt3_img = tgt3_img.cuda()

            # training G
            optimizer_trans.zero_grad()
            optimizer_in_src.zero_grad()
            optimizer_in_tgt1.zero_grad()
            optimizer_in_tgt2.zero_grad()
            optimizer_in_tgt3.zero_grad()
            adjust_learning_rate(optimizer_trans,args.learning_rate,global_step,num_steps)
            adjust_learning_rate(optimizer_in_src,args.learning_rate,global_step,num_steps)
            adjust_learning_rate(optimizer_in_tgt1,args.learning_rate,global_step,num_steps)
            adjust_learning_rate(optimizer_in_tgt2,args.learning_rate,global_step,num_steps)
            adjust_learning_rate(optimizer_in_tgt3,args.learning_rate,global_step,num_steps)

            for param in D_net_src.parameters():
                param.requires_grad = False       
            for param in D_net_tgt1.parameters():
                param.requires_grad = False     
            for param in D_net_tgt2.parameters():
                param.requires_grad = False     
            for param in D_net_tgt3.parameters():
                param.requires_grad = False          
            for param in trans_net.parameters():
                param.requires_grad = True
            for param in customizedin_src.parameters():
                param.requires_grad = True
            for param in customizedin_tgt1.parameters():
                param.requires_grad = True
            for param in customizedin_tgt2.parameters():
                param.requires_grad = True
            for param in customizedin_tgt3.parameters():
                param.requires_grad = True

            # reconstructed source image      
            src_recons = trans_net(src_img,customizedin_src)

            # reconstructed target image      
            tgt1_recons = trans_net(tgt1_img,customizedin_tgt1)
            tgt2_recons = trans_net(tgt2_img,customizedin_tgt2)
            tgt3_recons = trans_net(tgt3_img,customizedin_tgt3)

            # transferred image from target to source     
            tgt1_2src = trans_net(tgt1_img,customizedin_src)
            tgt2_2src = trans_net(tgt2_img,customizedin_src)
            tgt3_2src = trans_net(tgt3_img,customizedin_src)
           
            # transferred image to target1     
            tgt2_2tgt1 = trans_net(tgt2_img,customizedin_tgt1)
            tgt3_2tgt1 = trans_net(tgt3_img,customizedin_tgt1)
            src_2tgt1 = trans_net(src_img,customizedin_tgt1)
            
            # transferred image to target2     
            tgt1_2tgt2 = trans_net(tgt1_img,customizedin_tgt2)
            tgt3_2tgt2 = trans_net(tgt3_img,customizedin_tgt2)
            src_2tgt2 = trans_net(src_img,customizedin_tgt2)
            
            # transferred image to target3     
            tgt1_2tgt3 = trans_net(tgt1_img,customizedin_tgt3)
            tgt2_2tgt3 = trans_net(tgt2_img,customizedin_tgt3)
            src_2tgt3 = trans_net(src_img,customizedin_tgt3)

            # reconstruction loss
            recons_loss_src = recons_loss(src_recons,src_img)
            recons_loss_tgt = recons_loss(tgt1_recons,tgt1_img) + recons_loss(tgt2_recons,tgt2_img) + recons_loss(tgt3_recons,tgt3_img)
            
            # adversarial loss
            D_tgt1_2src = D_net_src(tgt1_2src)
            Label_tgt1_2src = Variable(torch.FloatTensor(D_tgt1_2src.data.size()).fill_(real_label)).cuda()
            D_tgt2_2src = D_net_src(tgt2_2src)
            Label_tgt2_2src = Variable(torch.FloatTensor(D_tgt2_2src.data.size()).fill_(real_label)).cuda()
            D_tgt3_2src = D_net_src(tgt3_2src)
            Label_tgt3_2src = Variable(torch.FloatTensor(D_tgt3_2src.data.size()).fill_(real_label)).cuda()

            D_tgt2_2tgt1 = D_net_tgt1(tgt2_2tgt1)
            Label_tgt2_2tgt1 = Variable(torch.FloatTensor(D_tgt2_2tgt1.data.size()).fill_(real_label)).cuda()
            D_tgt3_2tgt1 = D_net_tgt1(tgt3_2tgt1)
            Label_tgt3_2tgt1 = Variable(torch.FloatTensor(D_tgt3_2tgt1.data.size()).fill_(real_label)).cuda()
            D_src_2tgt1 = D_net_tgt1(src_2tgt1)
            Label_src_2tgt1 = Variable(torch.FloatTensor(D_src_2tgt1.data.size()).fill_(real_label)).cuda()

            D_tgt1_2tgt2 = D_net_tgt2(tgt1_2tgt2)
            Label_tgt1_2tgt2 = Variable(torch.FloatTensor(D_tgt1_2tgt2.data.size()).fill_(real_label)).cuda()
            D_tgt3_2tgt2 = D_net_tgt2(tgt3_2tgt2)
            Label_tgt3_2tgt2 = Variable(torch.FloatTensor(D_tgt3_2tgt2.data.size()).fill_(real_label)).cuda()
            D_src_2tgt2 = D_net_tgt2(src_2tgt2)
            Label_src_2tgt2 = Variable(torch.FloatTensor(D_src_2tgt2.data.size()).fill_(real_label)).cuda()
            
            D_tgt1_2tgt3 = D_net_tgt3(tgt1_2tgt3)
            Label_tgt1_2tgt3 = Variable(torch.FloatTensor(D_tgt1_2tgt3.data.size()).fill_(real_label)).cuda()
            D_tgt2_2tgt3 = D_net_tgt3(tgt2_2tgt3)
            Label_tgt2_2tgt3 = Variable(torch.FloatTensor(D_tgt2_2tgt3.data.size()).fill_(real_label)).cuda()
            D_src_2tgt3 = D_net_tgt3(src_2tgt3)
            Label_src_2tgt3 = Variable(torch.FloatTensor(D_src_2tgt3.data.size()).fill_(real_label)).cuda()

            loss_adv_G_src = bce_loss(D_tgt1_2src,Label_tgt1_2src) + bce_loss(D_tgt2_2src,Label_tgt2_2src) + bce_loss(D_tgt3_2src,Label_tgt3_2src) \
                           + bce_loss(D_tgt2_2tgt1,Label_tgt2_2tgt1) + bce_loss(D_tgt3_2tgt1,Label_tgt3_2tgt1) + bce_loss(D_src_2tgt1,Label_src_2tgt1) \
                           + bce_loss(D_tgt1_2tgt2,Label_tgt1_2tgt2) + bce_loss(D_tgt3_2tgt2,Label_tgt3_2tgt2) + bce_loss(D_src_2tgt2,Label_src_2tgt2) \
                           + bce_loss(D_tgt1_2tgt3,Label_tgt1_2tgt3) + bce_loss(D_tgt2_2tgt3,Label_tgt2_2tgt3) + bce_loss(D_src_2tgt3,Label_src_2tgt3)     

            # complete loss       
            total_loss = (recons_loss_src + recons_loss_tgt) * args.rec_weight + loss_adv_G_src 
            
            loss_hist[global_step,0] = total_loss.item()
            loss_hist[global_step,1] = recons_loss_src.item() + recons_loss_tgt.item()
            loss_hist[global_step,2] = loss_adv_G_src.item()
   
            writer.add_scalar('total_loss_iter', total_loss.item(), global_step)
            writer.add_scalar('recons_loss_src_iter', recons_loss_src.item(), global_step)
            writer.add_scalar('recons_loss_tgt_iter', recons_loss_tgt.item(), global_step)
            writer.add_scalar('loss_adv_G_src_iter', loss_adv_G_src.item(), global_step)
            total_loss.backward()
            optimizer_trans.step()
            optimizer_in_src.step()
            optimizer_in_tgt1.step()
            optimizer_in_tgt2.step()
            optimizer_in_tgt3.step()

            # train D
            # bring back requires_grad
            for param in D_net_src.parameters():
                param.requires_grad = True        
            for param in D_net_tgt1.parameters():
                param.requires_grad = True     
            for param in D_net_tgt2.parameters():
                param.requires_grad = True     
            for param in D_net_tgt3.parameters():
                param.requires_grad = True         
            for param in trans_net.parameters():
                param.requires_grad = False
            for param in customizedin_src.parameters():
                param.requires_grad = False
            for param in customizedin_tgt1.parameters():
                param.requires_grad = False
            for param in customizedin_tgt2.parameters():
                param.requires_grad = False
            for param in customizedin_tgt3.parameters():
                param.requires_grad = False
            
            optimizer_D_src.zero_grad()
            adjust_learning_rate(optimizer_D_src,args.learning_rate_D,global_step,num_steps)       
            optimizer_D_tgt1.zero_grad()
            adjust_learning_rate(optimizer_D_tgt1,args.learning_rate_D,global_step,num_steps)     
            optimizer_D_tgt2.zero_grad()
            adjust_learning_rate(optimizer_D_tgt2,args.learning_rate_D,global_step,num_steps)     
            optimizer_D_tgt3.zero_grad()
            adjust_learning_rate(optimizer_D_tgt3,args.learning_rate_D,global_step,num_steps)       
            
            # transferred image from target to source
            tgt1_2src = trans_net(tgt1_img,customizedin_src)
            tgt2_2src = trans_net(tgt2_img,customizedin_src)
            tgt3_2src = trans_net(tgt3_img,customizedin_src)

            # transferred image to target1     
            tgt2_2tgt1 = trans_net(tgt2_img,customizedin_tgt1)
            tgt3_2tgt1 = trans_net(tgt3_img,customizedin_tgt1)
            src_2tgt1 = trans_net(src_img,customizedin_tgt1)
            
            # transferred image to target2     
            tgt1_2tgt2 = trans_net(tgt1_img,customizedin_tgt2)
            tgt3_2tgt2 = trans_net(tgt3_img,customizedin_tgt2)
            src_2tgt2 = trans_net(src_img,customizedin_tgt2)
            
            # transferred image to target3     
            tgt1_2tgt3 = trans_net(tgt1_img,customizedin_tgt3)
            tgt2_2tgt3 = trans_net(tgt2_img,customizedin_tgt3)
            src_2tgt3 = trans_net(src_img,customizedin_tgt3)
                      
            D_tgt1_2src = D_net_src(tgt1_2src)     
            Label_tgt1_2src = Variable(torch.FloatTensor(D_tgt1_2src.data.size()).fill_(fake_label)).cuda() 
            D_tgt2_2src = D_net_src(tgt2_2src)     
            Label_tgt2_2src = Variable(torch.FloatTensor(D_tgt2_2src.data.size()).fill_(fake_label)).cuda() 
            D_tgt3_2src = D_net_src(tgt3_2src)     
            Label_tgt3_2src = Variable(torch.FloatTensor(D_tgt3_2src.data.size()).fill_(fake_label)).cuda() 
            D_src = D_net_src(src_img)
            Label_src = Variable(torch.FloatTensor(D_src.data.size()).fill_(real_label)).cuda()

            D_tgt2_2tgt1 = D_net_tgt1(tgt2_2tgt1)     
            Label_tgt2_2tgt1 = Variable(torch.FloatTensor(D_tgt2_2tgt1.data.size()).fill_(fake_label)).cuda() 
            D_tgt3_2tgt1 = D_net_tgt1(tgt3_2tgt1)     
            Label_tgt3_2tgt1 = Variable(torch.FloatTensor(D_tgt3_2tgt1.data.size()).fill_(fake_label)).cuda() 
            D_scr_2tgt1 = D_net_tgt1(src_2tgt1)     
            Label_scr_2tgt1 = Variable(torch.FloatTensor(D_scr_2tgt1.data.size()).fill_(fake_label)).cuda() 
            D_tgt1 = D_net_tgt1(tgt1_img)
            Label_tgt1 = Variable(torch.FloatTensor(D_tgt1.data.size()).fill_(real_label)).cuda()
        
            D_tgt1_2tgt2 = D_net_tgt2(tgt1_2tgt2)     
            Label_tgt1_2tgt2 = Variable(torch.FloatTensor(D_tgt1_2tgt2.data.size()).fill_(fake_label)).cuda() 
            D_tgt3_2tgt2 = D_net_tgt2(tgt3_2tgt2)     
            Label_tgt3_2tgt2 = Variable(torch.FloatTensor(D_tgt3_2tgt2.data.size()).fill_(fake_label)).cuda() 
            D_scr_2tgt2 = D_net_tgt2(src_2tgt2)     
            Label_scr_2tgt2 = Variable(torch.FloatTensor(D_scr_2tgt2.data.size()).fill_(fake_label)).cuda() 
            D_tgt2 = D_net_tgt2(tgt2_img)
            Label_tgt2 = Variable(torch.FloatTensor(D_tgt2.data.size()).fill_(real_label)).cuda()

            D_tgt1_2tgt3 = D_net_tgt3(tgt1_2tgt3)     
            Label_tgt1_2tgt3 = Variable(torch.FloatTensor(D_tgt1_2tgt3.data.size()).fill_(fake_label)).cuda() 
            D_tgt2_2tgt3 = D_net_tgt3(tgt2_2tgt3)     
            Label_tgt2_2tgt3 = Variable(torch.FloatTensor(D_tgt2_2tgt3.data.size()).fill_(fake_label)).cuda() 
            D_scr_2tgt3 = D_net_tgt3(src_2tgt3)     
            Label_scr_2tgt3 = Variable(torch.FloatTensor(D_scr_2tgt3.data.size()).fill_(fake_label)).cuda() 
            D_tgt3 = D_net_tgt3(tgt3_img)
            Label_tgt3 = Variable(torch.FloatTensor(D_tgt3.data.size()).fill_(real_label)).cuda()

            loss_adv_D_src = bce_loss(D_tgt1_2src,Label_tgt1_2src) + bce_loss(D_tgt2_2src,Label_tgt2_2src) + bce_loss(D_tgt3_2src,Label_tgt3_2src) + bce_loss(D_src,Label_src) \
                           + bce_loss(D_tgt2_2tgt1,Label_tgt2_2tgt1) + bce_loss(D_tgt3_2tgt1,Label_tgt3_2tgt1) + bce_loss(D_scr_2tgt1,Label_scr_2tgt1) + bce_loss(D_tgt1,Label_tgt1) \
                           + bce_loss(D_tgt1_2tgt2,Label_tgt1_2tgt2) + bce_loss(D_tgt3_2tgt2,Label_tgt3_2tgt2) + bce_loss(D_scr_2tgt2,Label_scr_2tgt2) + bce_loss(D_tgt2,Label_tgt2) \
                           + bce_loss(D_tgt1_2tgt3,Label_tgt1_2tgt3) + bce_loss(D_tgt2_2tgt3,Label_tgt2_2tgt3) + bce_loss(D_scr_2tgt3,Label_scr_2tgt3) + bce_loss(D_tgt3,Label_tgt3) 
            
            loss_adv_D_src.backward()
            optimizer_D_src.step()
            optimizer_D_tgt1.step()
            optimizer_D_tgt2.step()
            optimizer_D_tgt3.step()

            loss_hist[global_step,6] = loss_adv_D_src.item()
            writer.add_scalar('loss_adv_D_src_iter', loss_adv_D_src.item(), global_step)

            batch_time = time.time()-tem_time
        
            if (batch_index+1) % 10 == 0: 
                print('epoch %d/%d:  %d/%d time: %.2fs recons_loss = %.3f G_loss = %.3f D_loss = %.3f \n'\
                    %(epoch+1, args.num_epoch,batch_index+1,num_batches,batch_time*10,np.mean(loss_hist[global_step-9:global_step+1,1])*100,
                    np.mean(loss_hist[global_step-9:global_step+1,2])*100,np.mean(loss_hist[global_step-9:global_step+1,6])*100))

            if (batch_index+1) % 1000 == 0: 
                vis_src_img = recreate_image(src_img[0].cpu())  
                vis_src_recons = recreate_image(src_recons[0].cpu())     
                vis_src_2tgt1 = recreate_image(src_2tgt1[0].cpu())    
                vis_src_2tgt2 = recreate_image(src_2tgt2[0].cpu())    
                vis_src_2tgt3 = recreate_image(src_2tgt3[0].cpu())              

                vis_tgt1_col = recreate_image(tgt1_img[0].cpu())    
                vis_tgt1_recons = recreate_image(tgt1_recons[0].cpu())    
                vis_tgt1_2src = recreate_image(tgt1_2src[0].cpu())   
                vis_tgt1_2tgt2 = recreate_image(tgt1_2tgt2[0].cpu())   
                vis_tgt1_2tgt3 = recreate_image(tgt1_2tgt3[0].cpu())   
                
                vis_tgt2_col = recreate_image(tgt2_img[0].cpu())    
                vis_tgt2_recons = recreate_image(tgt2_recons[0].cpu())    
                vis_tgt2_2src = recreate_image(tgt2_2src[0].cpu())    
                vis_tgt2_2tgt1 = recreate_image(tgt2_2tgt1[0].cpu())    
                vis_tgt2_2tgt3 = recreate_image(tgt2_2tgt3[0].cpu())  
                
                vis_tgt3_col = recreate_image(tgt3_img[0].cpu())    
                vis_tgt3_recons = recreate_image(tgt3_recons[0].cpu())    
                vis_tgt3_2src = recreate_image(tgt3_2src[0].cpu())    
                vis_tgt3_2tgt1 = recreate_image(tgt3_2tgt1[0].cpu())    
                vis_tgt3_2tgt2 = recreate_image(tgt3_2tgt2[0].cpu())   

                plt.subplot(4, 5, 1)
                plt.imshow(vis_src_img)
                plt.title('src_img')        
                plt.axis('off')
                plt.subplot(4, 5, 2)
                plt.imshow(vis_src_recons)
                plt.title('src_recons')        
                plt.axis('off')
                plt.subplot(4, 5, 3)
                plt.imshow(vis_src_2tgt1)
                plt.title('src_2tgt1')        
                plt.axis('off')
                plt.subplot(4, 5, 4)
                plt.imshow(vis_src_2tgt2)
                plt.title('src_2tgt2')        
                plt.axis('off')
                plt.subplot(4, 5, 5)
                plt.imshow(vis_src_2tgt3)
                plt.title('src_2tgt3')        
                plt.axis('off')
                         
                plt.subplot(4, 5, 6)
                plt.imshow(vis_tgt1_col)
                plt.title('tgt1_img')        
                plt.axis('off')
                plt.subplot(4, 5, 7)
                plt.imshow(vis_tgt1_recons)
                plt.title('tgt1_recons')        
                plt.axis('off')
                plt.subplot(4, 5, 8)
                plt.imshow(vis_tgt1_2src)
                plt.title('tgt1_2src')        
                plt.axis('off')
                plt.subplot(4, 5, 9)
                plt.imshow(vis_tgt1_2tgt2)
                plt.title('tgt1_2tgt2')        
                plt.axis('off')
                plt.subplot(4, 5, 10)
                plt.imshow(vis_tgt1_2tgt3)
                plt.title('tgt1_2tgt3')        
                plt.axis('off')
                
                plt.subplot(4, 5, 11)
                plt.imshow(vis_tgt2_col)
                plt.title('tgt2_img')        
                plt.axis('off')
                plt.subplot(4, 5, 12)
                plt.imshow(vis_tgt2_recons)
                plt.title('tgt2_recons')        
                plt.axis('off')
                plt.subplot(4, 5, 13)
                plt.imshow(vis_tgt2_2src)
                plt.title('tgt2_2src')        
                plt.axis('off')
                plt.subplot(4, 5, 14)
                plt.imshow(vis_tgt2_2tgt1)
                plt.title('tgt2_2tgt1')        
                plt.axis('off')
                plt.subplot(4, 5, 15)
                plt.imshow(vis_tgt2_2tgt3)
                plt.title('tgt2_2tgt3')        
                plt.axis('off')
                
                plt.subplot(4, 5, 16)
                plt.imshow(vis_tgt3_col)
                plt.title('tgt3_img')        
                plt.axis('off')
                plt.subplot(4, 5, 17)
                plt.imshow(vis_tgt3_recons)
                plt.title('tgt3_recons')        
                plt.axis('off')
                plt.subplot(4, 5, 18)
                plt.imshow(vis_tgt3_2src)
                plt.title('tgt3_2src')        
                plt.axis('off')
                plt.subplot(4, 5, 19)
                plt.imshow(vis_tgt3_2tgt1)
                plt.title('tgt3_2tgt1')        
                plt.axis('off')
                plt.subplot(4, 5, 20)
                plt.imshow(vis_tgt3_2tgt2)
                plt.title('tgt3_2tgt2')        
                plt.axis('off')

                writer.add_figure('color', plt.gcf(), global_step)
            global_step += 1

        im = vis_src_recons.resize(original_size_src, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_src_recons_'+src_name[0],'png')   
        im = vis_src_2tgt1.resize(original_size_src, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_src_2tgt1_'+src_name[0],'png')   
        im = vis_src_2tgt2.resize(original_size_src, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_src_2tgt2_'+src_name[0],'png')   
        im = vis_src_2tgt3.resize(original_size_src, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_src_2tgt3_'+src_name[0],'png')   

        im = vis_tgt1_recons.resize(original_size_tgt1, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt1_recons_'+tgt1_name[0],'png')  
        im = vis_tgt1_2src.resize(original_size_tgt1, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt1_2src_'+tgt1_name[0],'png')  
        im = vis_tgt1_2tgt2.resize(original_size_tgt1, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt1_2tgt2_'+tgt1_name[0],'png') 
        im = vis_tgt1_2tgt3.resize(original_size_tgt1, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt1_2tgt3_'+tgt1_name[0],'png') 
        
        im = vis_tgt2_recons.resize(original_size_tgt2, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt2_recons_'+tgt2_name[0],'png')  
        im = vis_tgt2_2src.resize(original_size_tgt2, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt2_2src_'+tgt2_name[0],'png')  
        im = vis_tgt2_2tgt1.resize(original_size_tgt2, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt2_2tgt1_'+tgt2_name[0],'png')  
        im = vis_tgt2_2tgt3.resize(original_size_tgt2, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt2_2tgt3_'+tgt2_name[0],'png')  
        
        im = vis_tgt3_recons.resize(original_size_tgt3, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt3_recons_'+tgt3_name[0],'png')  
        im = vis_tgt3_2src.resize(original_size_tgt3, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt3_2src_'+tgt3_name[0],'png')  
        im = vis_tgt3_2tgt1.resize(original_size_tgt3, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt3_2tgt1_'+tgt3_name[0],'png')  
        im = vis_tgt3_2tgt2.resize(original_size_tgt3, Image.BICUBIC)       
        im.save(snapshot_dir+'epoch_'+repr(epoch+1)+'_tgt3_2tgt2_'+tgt3_name[0],'png')  

        # Saving the models 
        print('Save Model')                     
        model_name = 'GTA_Trans_epoch_'+repr(epoch+1)+'_batch_'+repr(batch_index+1)+'.pth'
        torch.save(trans_net.state_dict(), os.path.join(
            snapshot_dir, model_name))        
        model_name = 'GTA_CIN_src_epoch_'+repr(epoch+1)+'_batch_'+repr(batch_index+1)+'.pth'
        torch.save(customizedin_src.state_dict(), os.path.join(
            snapshot_dir, model_name))    
        model_name = 'GTA_CIN_tgt1_epoch_'+repr(epoch+1)+'_batch_'+repr(batch_index+1)+'.pth'
        torch.save(customizedin_tgt1.state_dict(), os.path.join(
            snapshot_dir, model_name))   
        model_name = 'GTA_CIN_tgt2_epoch_'+repr(epoch+1)+'_batch_'+repr(batch_index+1)+'.pth'
        torch.save(customizedin_tgt2.state_dict(), os.path.join(
            snapshot_dir, model_name))   
        model_name = 'GTA_CIN_tgt3_epoch_'+repr(epoch+1)+'_batch_'+repr(batch_index+1)+'.pth'
        torch.save(customizedin_tgt3.state_dict(), os.path.join(
            snapshot_dir, model_name))   
        model_name = 'GTA_D_src_epoch_'+repr(epoch+1)+'_batch_'+repr(batch_index+1)+'.pth'
        torch.save(D_net_src.state_dict(), os.path.join(
            snapshot_dir, model_name))     
    
if __name__ == '__main__':
    main()
