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
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from dataset.mapillary_dataset import MapDataSet
from model.Networks import DeeplabMulti,Discriminator,StyleTransferNet,CustomizedInstanceNorm
import torch.backends.cudnn as cudnn
import mask_gen

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
    parser.add_argument("--data_ID", type=int, default=0,
                        help="target dataset ID. 0: Cityscapes 1: IDD 2: Mapillary")
    parser.add_argument("--data_dir_source", type=str, default='/iarai/home/yonghao.xu/Data/SegmentationData/GTA5/',
                        help="source dataset path.")
    parser.add_argument("--data_list_source", type=str, default='./dataset/GTA5_imagelist_train.txt',
                        help="source dataset list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    parser.add_argument("--input_size_src", type=str, default='1280,720',
                        help="width and height of input source images.")    
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
    parser.add_argument("--T_restore_from", type=str, default='./pretrained_model/GTA_Trans_epoch_20_batch_6241.pth',
                        help="pretrained style transfer network T.")
    parser.add_argument("--source_IN", type=str, default='./pretrained_model/GTA_CIN_src_epoch_20_batch_6241.pth',
                        help="pretrained GTA style.")
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="regularisation parameter for L2-loss.")

    #hyperparameters
    parser.add_argument("--teacher_alpha", type=float, default=0.999,
                        help="teacher alpha in EMA.")
    parser.add_argument("--con_weight", type=float, default=100,
                        help="consistency regularization weight.")
    parser.add_argument("--out_weight", type=float, default=1e-3,
                        help="adversarial training weight.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./MT-KD/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

class seg_criterion(object):
    def __init__(self, num_classes,often_balance):
        self.num_classes = num_classes
        self.often_balance = often_balance
        self.class_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.often_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.max_value = 7

    def update_class_criterion(self,labels):
        weight = torch.FloatTensor(self.num_classes).zero_().cuda()
        weight += 1
        count = torch.FloatTensor(self.num_classes).zero_().cuda()
        often = torch.FloatTensor(self.num_classes).zero_().cuda()
        often += 1
        
        n, h, w = labels.shape
        for i in range(self.num_classes):
            count[i] = torch.sum(labels==i)
            if count[i] < 64*64*n:
                weight[i] = self.max_value
        if self.often_balance:
            often[count == 0] = self.max_value

        self.often_weight = 0.9 * self.often_weight + 0.1 * often 
        self.class_weight = weight * self.often_weight
        
        return nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255)

def main():

    args = get_arguments()
    w, h = map(int, args.input_size_src.split(','))
    input_size_src = (w, h)
    w, h = map(int, args.input_size_tgt.split(','))
    input_size_tgt = (w, h)

    src_loader = data.DataLoader(
                    GTA5DataSet(args.data_dir_source, args.data_list_source, max_iters=int(np.ceil(args.num_steps_stop*args.batch_size)),
                    crop_size=input_size_src,
                    scale=False, mirror=False, mean=IMG_MEAN,num_classes=args.num_classes),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if args.data_ID == 0:
        # G->C
        data_list_target = './dataset/cityscapes_labellist_train.txt'
        data_list_val = './dataset/cityscapes_labellist_val.txt'
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/Cityscapes/'     
        input_size_test = (2048,1024)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2City/MTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
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

    elif args.data_ID == 1:
        # G->I
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/IDD/IDD_Segmentation/'
        data_list_val = './dataset/IDD_val.txt'
        data_list_target = './dataset/IDD_train.txt'
        input_size_test = (1920,1080)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2IDD/MTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
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
        # G->M
        data_dir_target = '/iarai/home/yonghao.xu/Data/SegmentationData/MapillaryVistas/'
        data_list_val = './dataset/Mapillary_val.txt'
        data_list_target = './dataset/Mapillary_train.txt'
        input_size_test = (1920,1080)

        snapshot_dir = args.snapshot_dir+'synthetic2real_GTA2Mapillary/MTKD_out_weight_'+str(args.out_weight)+'_con_weight_'+str(args.con_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
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
    trans_net = StyleTransferNet(input_channel=3,output_channel=3)
    
    saved_state_dict = torch.load(args.T_restore_from,map_location='cuda:0')

    new_params = trans_net.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        new_params[j] = saved_state_dict[i]
    trans_net.load_state_dict(new_params)

    for name, param in trans_net.named_parameters():
        param.requires_grad=False
    trans_net = trans_net.cuda()
    
    customizedin_gta = CustomizedInstanceNorm()    
    saved_state_dict = torch.load(args.source_IN,map_location='cuda:0')
    new_params = customizedin_gta.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        new_params[j] = saved_state_dict[i]
    customizedin_gta.load_state_dict(new_params)
    for name, param in customizedin_gta.named_parameters():
        param.requires_grad=False
    customizedin_gta = customizedin_gta.cuda()

    student_net = DeeplabMulti(num_classes=args.num_classes)
    teacher_net = DeeplabMulti(num_classes=args.num_classes)
    D_net1 = Discriminator(input_channel=args.num_classes)
    D_net2 = Discriminator(input_channel=args.num_classes)
    D_net1.train()
    D_net2.train()
        
    saved_state_dict = torch.load(args.restore_from)   
    
    student_net.load_state_dict(saved_state_dict)
    teacher_net.load_state_dict(saved_state_dict)    
    
    for name, param in teacher_net.named_parameters():    
        param.requires_grad=False

    student_net = student_net.cuda()
    teacher_net = teacher_net.cuda()
    D_net1 = D_net1.cuda()
    D_net2 = D_net2.cuda()
   
    optimizer = optim.SGD(student_net.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    optimizer_D1 = optim.Adam(D_net1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    optimizer_D2 = optim.Adam(D_net2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()
    
    student_params = list(student_net.parameters())
    teacher_params = list(teacher_net.parameters())

    teacher_optimizer = WeightEMA(
        teacher_params, 
        student_params,
        alpha=args.teacher_alpha,
    )

    interp_src = nn.Upsample(size=(input_size_src[1], input_size_src[0]),  mode='bilinear', align_corners=True)
    interp_tgt = nn.Upsample(size=(input_size_tgt[1], input_size_tgt[0]),  mode='bilinear', align_corners=True)
    n_class = args.num_classes
    loss_hist = np.zeros((args.num_steps_stop,7))

    mIoU_hist = 20
    seg_crit = seg_criterion(args.num_classes,often_balance=True)
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
    
    for batch_index, (src_data, tgt_data) in enumerate(zip(src_loader, tgt_loader)):
        if batch_index>=args.num_steps_stop:
            break
        
        decay_adv = (1 - batch_index/args.num_steps)
        tem_time = time.time()
        optimizer.zero_grad()
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate(optimizer,args.learning_rate,batch_index,args.num_steps)
        adjust_learning_rate(optimizer_D1,args.learning_rate_D,batch_index,args.num_steps)
        adjust_learning_rate(optimizer_D2,args.learning_rate_D,batch_index,args.num_steps)

        # Train F_S
        for param in student_net.parameters():
            param.requires_grad = True
        for param in D_net1.parameters():
            param.requires_grad = False
        for param in D_net2.parameters():
            param.requires_grad = False
        
        src_s_input, src_label, _, _ = src_data
        src_s_input = src_s_input.cuda()
        src_label = src_label.long().cuda()            
        src_output_ori1,src_output_ori2 = student_net(src_s_input)
        src_output1 = interp_src(src_output_ori1)
        src_output2 = interp_src(src_output_ori2)

        # L_CE
        seg_loss = seg_crit.update_class_criterion(src_label)
        cls_loss_value = seg_loss(src_output1, src_label)*0.1 + seg_loss(src_output2, src_label)
        _, predict_labels = torch.max(src_output2, 1)
        lbl_pred = predict_labels.detach().cpu().numpy()
        lbl_true = src_label.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
            metrics_batch.append(mean_iu)                
        miou = np.mean(metrics_batch, axis=0)  
        
        # L_con
        images, _,_, _ = tgt_data
        images = images.cuda()
        tgt2src = trans_net(images,customizedin_gta)   

        tgt2src_1 = tgt2src[:half_batch,:,:,:]
        images_1 = images[:half_batch,:,:,:]
        images_2 = images[half_batch:,:,:,:]
        
        mask_size = tgt2src.shape[2:]
        # (N,1,H,W)
        batch_mix_masks = torch.from_numpy(mask_generator.generate_params(len(tgt2src_1), mask_size)).cuda().float()
            
        # Mix images with masks
        mixed_s = tgt2src_1 * (1 - batch_mix_masks) + images_2 * batch_mix_masks
       
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
        total_loss = cls_loss_value + args.con_weight * con_loss_value + args.out_weight * loss_adv_G * decay_adv            
    
        loss_hist[batch_index,0] = total_loss.item()
        loss_hist[batch_index,1] = cls_loss_value.item()
        loss_hist[batch_index,2] = con_loss_value.item()
        loss_hist[batch_index,3] = loss_adv_G.item()
        loss_hist[batch_index,6] = miou

        total_loss.backward()

        # train D_out
        for param in D_net1.parameters():
            param.requires_grad = True
        for param in D_net2.parameters():
            param.requires_grad = True
        for param in student_net.parameters():
            param.requires_grad = False

        # L_out
        # train with source
        pred1 = src_output1.detach()
        pred2 = src_output2.detach()
    
        D_out1 = D_net1(F.softmax(pred1, dim=1))
        D_out2 = D_net2(F.softmax(pred2, dim=1))

        loss_D1 = bce_loss(D_out1,
                            Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())*0.5

        loss_D2 = bce_loss(D_out2,
                            Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())*0.5

        loss_D1.backward()
        loss_D2.backward()

        loss_hist[batch_index,4] = loss_D1.data.cpu().numpy()
        loss_hist[batch_index,5] = loss_D2.data.cpu().numpy()

        # train with target
        pred_target1 = tgt_s_output1.detach()
        pred_target2 = tgt_s_output1.detach()

        D_out1 = D_net1(F.softmax(pred_target1, dim=1))
        D_out2 = D_net2(F.softmax(pred_target2, dim=1))

        loss_D1 = bce_loss(D_out1,
                            Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())*0.5

        loss_D2 = bce_loss(D_out2,
                            Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())*0.5

        loss_D1.backward()
        loss_D2.backward()

        loss_hist[batch_index,4] += loss_D1.data.cpu().numpy()
        loss_hist[batch_index,5] += loss_D2.data.cpu().numpy()

        optimizer.step()
        teacher_optimizer.step()   
        optimizer_D1.step() 
        optimizer_D2.step()

        batch_time = time.time()-tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d time: %.2f miou = %.1f cls_loss = %.3f con_loss = %.3f g_loss = %.3f d1_loss = %.3f d2_loss = %.3f \n'\
                %(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,6])*100,\
                    np.mean(loss_hist[batch_index-9:batch_index+1,1]),np.mean(loss_hist[batch_index-9:batch_index+1,2]),\
                        np.mean(loss_hist[batch_index-9:batch_index+1,3]),np.mean(loss_hist[batch_index-9:batch_index+1,4]),np.mean(loss_hist[batch_index-9:batch_index+1,5])))
            f.write('Iter %d/%d time: %.2f miou = %.1f cls_loss = %.3f con_loss = %.3f g_loss = %.3f d1_loss = %.3f d2_loss = %.3f \n'\
                %(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,6])*100,\
                    np.mean(loss_hist[batch_index-9:batch_index+1,1]),np.mean(loss_hist[batch_index-9:batch_index+1,2]),\
                        np.mean(loss_hist[batch_index-9:batch_index+1,3]),np.mean(loss_hist[batch_index-9:batch_index+1,4]),np.mean(loss_hist[batch_index-9:batch_index+1,5])))
            f.flush() 
            
        if (batch_index+1) % 500 == 0:                    
            mIoU_new = test_mIoU7(f,teacher_net, val_loader, batch_index+1,input_size_test,print_per_batches=10)
            
            # Saving the models        
            if mIoU_new > mIoU_hist:    
                f.write('Save Model\n') 
                print('Save Model')                     
                model_name = 'Seg_batch_'+repr(batch_index+1)+'_miou_'+repr(int(mIoU_new*100))+'.pth'
                torch.save(teacher_net.state_dict(), os.path.join(
                    snapshot_dir, model_name))   
                model_name = 'D_net1_batch_'+repr(batch_index+1)+'.pth'
                torch.save(D_net1.state_dict(), os.path.join(
                    snapshot_dir, model_name))   
                model_name = 'D_net2_batch_'+repr(batch_index+1)+'.pth'
                torch.save(D_net2.state_dict(), os.path.join(
                    snapshot_dir, model_name))      
                mIoU_hist = mIoU_new

    f.close()
    np.savez(snapshot_dir+'loss.npz',loss_hist=loss_hist) 
    
if __name__ == '__main__':
    main()
