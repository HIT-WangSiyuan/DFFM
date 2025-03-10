from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from Load_FAS_MultiModal_DropModal import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout,Spoofing_train_UUID
# from Load_FAS_MultiModal import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout
from Load_FAS_MultiModal_DropModal_test import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest,Spoofing_valtest_UUID

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from DFFM import Dynamic_Feature_Fusion_Model
import warnings
from utils_FAS_MultiModal import AvgrageMeter, performances_FAS_MultiModal,performances_FAS_MultiModal_C_W,performances_FAS_MultiMoodal_intra


warnings.filterwarnings("ignore")
##########    Dataset root    ##########

# root_dir    CASIA_SURF; CeFA; WMCA; PADISI;
root_FAS_dir = 'FAS_Data'

#three data
CPS_train='Protocols/CPS_train.txt'
CPW_train='Protocols/CPW_train.txt'
CSW_train='Protocols/CSW_train.txt'
PSW_train='Protocols/PSW_train.txt'
CPS_val='Protocols/CPS_val.txt'
CPW_val='Protocols/CPW_val.txt'
CSW_val='Protocols/CSW_val.txt'
PSW_val='Protocols/PSW_val.txt'
#tow data
CW_train='Protocols/CW_train.txt'
CW_val='Protocols/CW_val.txt'
CW_test='Protocols/CW_test.txt'
PS_train='Protocols/PS_train.txt'
PS_val='Protocols/PS_val.txt'
PS_test='Protocols/PS_test.txt'
#one data
CeFA_test='Protocols/CASIA-CeFA_test.txt'
SURF_test='Protocols/CASIA-SURF_test.txt'
WMCA_test='Protocols/WMCA_test.txt'
PADISI_test='Protocols/PADISI_test.txt'



class ContrastLoss(nn.Module):
    
    def __init__(self):
        super(ContrastLoss, self).__init__()
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        contrast_label = contrast_label.float()
        anchor_fea = anchor_fea.detach()
        loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
        loss = loss*contrast_label
        return loss.mean()

# main function

def train_test_protocol(train_file,test_file,val_file,name):
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log + '/' + args.log + '_'+name+'_'+'_log.txt', 'a')

    echo_batches = args.echo_batches

    print(f'finetune!\n')
    log_file.write(f'finetune!\n')
    log_file.flush()

    model = Dynamic_Feature_Fusion_Model()

    model = model.cuda()

    lr = args.lr
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    #print(model)

    binary_fuc = nn.CrossEntropyLoss()
    map_fuc = nn.MSELoss()
    contra_fun = ContrastLoss()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_absolute = AvgrageMeter()
        loss_contra = AvgrageMeter()
        loss_absolute_RGB = AvgrageMeter()

        ###########################################
        '''               train           '''
        ###########################################
        model.train()

        train_data = Spoofing_train_UUID(train_file, root_FAS_dir, UUID=0,transform=transforms.Compose(
            [RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs = sample_batched['image_x'].cuda()
            
            spoof_label = sample_batched['spoofing_label'].cuda()
            binary_mask = sample_batched['map_x1'].cuda()
            inputs_depth = sample_batched['image_x_depth'].cuda()
            inputs_ir = sample_batched['image_x_ir'].cuda()
            UUID=sample_batched['UUID'].cuda()
            m1_t=torch.zeros(inputs.shape[0]).cuda()
            m2_t=torch.ones(inputs.shape[0]).cuda()
            m3_t=(torch.ones(inputs.shape[0])*2).cuda()
            logits,m1,m2,m3 = model(inputs, inputs_depth, inputs_ir)
            
            
            optimizer.zero_grad()
            adv_loss = 0.3*binary_fuc(m1, m1_t.long())+0.3*binary_fuc(m2, m2_t.long())+0.3*binary_fuc(m3, m3_t.long())
            binary_loss=binary_fuc(logits, spoof_label.squeeze(-1))
            loss = binary_loss

            loss.backward()
            optimizer.step()

            n = inputs.size(0)
            loss_absolute.update(loss.data, n)

            if i % echo_batches == echo_batches - 1:  # print every 50 mini-batches

                # visualization

                # log written
                print(
                    'epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f  \n' % (epoch + 1, i + 1, lr, loss_absolute.avg))

        # whole epoch average
        log_file.write(
            'epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f  \n' % (epoch + 1, i + 1, lr, loss_absolute.avg))
        log_file.flush()

        #### validation/test

        epoch_test = 5
        if epoch % epoch_test == epoch_test - 1:  # test every 5 epochs
            model.eval()

            with torch.no_grad():
                val_data = Spoofing_valtest(val_file, root_FAS_dir,
                                             transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_val = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=4)
                test_data = Spoofing_valtest(test_file, root_FAS_dir,
                                             transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                ###############################################################################################
                '''                                            ALL                            '''
                ##############################################################################################

                ###########################################
                '''                val             '''
                ##########################################
                # val
                

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_val):

                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
                    image_x_zeros = sample_batched['image_x_zeros'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()
                    logits,m1,m2,m3 = model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))

                val_filename = args.log + '/' + args.log + '_'+name+'_val.txt'
                with open(val_filename, 'w') as file:
                    file.writelines(map_score_list)

                ###########################################
                '''                test             '''
                ##########################################
                # Intra-test for CASIA_SURF
                

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_test):
                    # log_file.write('test SiW i= %d \n' % (i))

                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
                    image_x_zeros = sample_batched['image_x_zeros'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()
                    logits,m1,m2,m3 = model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))

                test_filename = args.log + '/' + args.log + '_'+name+'_test.txt'
                with open(test_filename, 'w') as file:
                    file.writelines(map_score_list)

                    ##########################################





                    ##########################################################################
                #       Performance measurement for both intra-testings
                ##########################################################################
                APCER,BPCER,ACER,AUC,TPR_FPR0001=performances_FAS_MultiMoodal_intra(val_filename,test_filename)

                print('\n\n test in %s(ALL): \n epoch:%d, Intra-testing!\n %s:  ACER= %.4f,APCER=%.4f,BPCER=%.4f, TPR_FPR0001= %.4f, AUC=%.4f' % (
                name,epoch + 1, name,ACER,APCER,BPCER, TPR_FPR0001,AUC))
                log_file.write(
                    '\n\n test in %s(ALL): \n epoch:%d, Intra-testing!\n %s:  ACER= %.4f,APCER=%.4f,BPCER=%.4f, TPR_FPR0001= %.4f, AUC=%.4f' % (
                name,epoch + 1, name,ACER,APCER,BPCER, TPR_FPR0001,AUC))
                log_file.flush()
                ###############################################################################################
                '''                                            Miss D                          '''
                ##############################################################################################

                ###########################################
                '''                val             '''
                ##########################################
                

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_val):
                    # log_file.write('test SiW i= %d \n' % (i))

                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
                    image_x_zeros = sample_batched['image_x_zeros'].cuda()
                    optimizer.zero_grad()

                    # pdb.set_trace()
                    logits,m1,m2,m3 = model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                val_filename = args.log + '/' + args.log + '_'+name+'_val.txt'
                with open(val_filename, 'w') as file:
                    file.writelines(map_score_list)

                ###########################################
                '''                test             '''
                ##########################################

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_test):
                    # log_file.write('test SiW i= %d \n' % (i))

                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
                    image_x_zeros = sample_batched['image_x_zeros'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()
                    logits,m1,m2,m3 = model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))

                test_filename = args.log + '/' + args.log + '_'+name+'_test.txt'
                with open(test_filename, 'w') as file:
                    file.writelines(map_score_list)

                    ##########################################





                    ##########################################################################
                #       Performance measurement for both intra- and inter-testings
                ##########################################################################
                APCER,BPCER,ACER,AUC,TPR_FPR0001=performances_FAS_MultiMoodal_intra(val_filename,test_filename)
    
                print('\n\n test in %s(Miss D): \n epoch:%d, Intra-testing!\n %s:  ACER= %.4f,APCER=%.4f,BPCER=%.4f, TPR_FPR0001= %.4f, AUC=%.4f' % (
                name,epoch + 1, name,ACER,APCER,BPCER, TPR_FPR0001,AUC))
                log_file.write(
                    '\n\n test in %s(Miss D): \n epoch:%d, Intra-testing!\n %s:  ACER= %.4f,APCER=%.4f,BPCER=%.4f, TPR_FPR0001= %.4f, AUC=%.4f' % (
                name,epoch + 1, name,ACER,APCER,BPCER, TPR_FPR0001,AUC))
                log_file.flush()

                ###############################################################################################
                '''                                            Miss I                            '''
                ##############################################################################################

                ###########################################
                '''                val             '''
                ##########################################


                map_score_list = []

                for i, sample_batched in enumerate(dataloader_val):
                    # log_file.write('test SiW i= %d \n' % (i))

                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
                    image_x_zeros = sample_batched['image_x_zeros'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()
                    logits,m1,m2,m3 = model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))

                val_filename = args.log + '/' + args.log + '_'+name+'_val.txt'
                with open(val_filename, 'w') as file:
                    file.writelines(map_score_list)

                ###########################################
                '''                test             '''
                ##########################################
                # Intra-test for CASIA_SURF

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_test):
                    # log_file.write('test SiW i= %d \n' % (i))

                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
                    image_x_zeros = sample_batched['image_x_zeros'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()
                    logits,m1,m2,m3 = model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))

                test_filename = args.log + '/' + args.log + '_'+name+'_test.txt'
                with open(test_filename, 'w') as file:
                    file.writelines(map_score_list)

                    ##########################################





                    ##########################################################################
                #       Performance measurement for both intra- and inter-testings
                ##########################################################################
                APCER,BPCER,ACER,AUC,TPR_FPR0001=performances_FAS_MultiMoodal_intra(val_filename,test_filename)

                print('\n\n test in %s(Miss I): \n epoch:%d, Intra-testing!\n %s:  ACER= %.4f,APCER=%.4f,BPCER=%.4f, TPR_FPR0001= %.4f, AUC=%.4f' % (
                name,epoch + 1, name,ACER,APCER,BPCER, TPR_FPR0001,AUC))
                log_file.write(
                    '\n\n test in %s(Miss I): \n epoch:%d, Intra-testing!\n %s:  ACER= %.4f,APCER=%.4f,BPCER=%.4f, TPR_FPR0001= %.4f, AUC=%.4f' % (
                name,epoch + 1, name,ACER,APCER,BPCER, TPR_FPR0001,AUC))
                log_file.flush()
                ###############################################################################################
                '''                                            Miss I&D                            '''
                ##############################################################################################

                ###########################################
                '''                val             '''
                ##########################################


                map_score_list = []

                for i, sample_batched in enumerate(dataloader_val):
                    # log_file.write('test SiW i= %d \n' % (i))

                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
                    image_x_zeros = sample_batched['image_x_zeros'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()
                    logits,m1,m2,m3= model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))

                val_filename = args.log + '/' + args.log + '_'+name+'_val.txt'
                with open(val_filename, 'w') as file:
                    file.writelines(map_score_list)

                ###########################################
                '''                test             '''
                ##########################################

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_test):
                    # log_file.write('test SiW i= %d \n' % (i))

                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
                    image_x_zeros = sample_batched['image_x_zeros'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()
                    logits,m1,m2,m3 = model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))

                test_filename = args.log + '/' + args.log + '_'+name+'_test.txt'
                with open(test_filename, 'w') as file:
                    file.writelines(map_score_list)

                    ##########################################





                    ##########################################################################
                #       Performance measurement for both intra- and inter-testings
                ##########################################################################
                APCER,BPCER,ACER,AUC,TPR_FPR0001=performances_FAS_MultiMoodal_intra(val_filename,test_filename)

                print('\n\n test in %s(Miss D&I): \n epoch:%d, Intra-testing!\n %s:  ACER= %.4f,APCER=%.4f,BPCER=%.4f, TPR_FPR0001= %.4f, AUC=%.4f' % (
                name,epoch + 1, name,ACER,APCER,BPCER, TPR_FPR0001,AUC))
                log_file.write(
                    '\n\n test in %s(Miss D&I): \n epoch:%d, Intra-testing!\n %s:  ACER= %.4f,APCER=%.4f,BPCER=%.4f, TPR_FPR0001= %.4f, AUC=%.4f' % (
                name,epoch + 1, name,ACER,APCER,BPCER, TPR_FPR0001,AUC))
                log_file.flush()
    
                

    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=100, help='how many batches display once')
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="DFFM",help='log and save model name')

    args = parser.parse_args()
#            3to1
    #CPS to W
    train_test_protocol(CPS_train,WMCA_test,CPS_val,"CPS2W")
    #CPW to S
    train_test_protocol(CPW_train,SURF_test,CPW_val,"CPW2S")
    #CSW to P
    train_test_protocol(CSW_train,PADISI_test,CSW_val,"CSW2P")
    #PSW to C
    train_test_protocol(PSW_train,CeFA_test,PSW_val,"PSW2C")
#            2to2
    #CW to PS  
    train_test_protocol(CW_train,PS_test,CW_val,"CW2PS")
    #PS to CW 
    train_test_protocol(PS_train,CW_test,PS_val,"PS2CW")