import os
import sys
import time
import random
import string
import argparse
import re

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from models import Model
from test_final import validation
from utils import get_args
import utils_dist as utils

import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import cv2
import torchvision.transforms as transforms

class WeightedBCELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super(WeightedBCELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, out, target, weights=None):
        out = out.clamp(self.epsilon, 1 - self.epsilon)
        if weights is not None:
            assert len(weights) == 2
            loss = weights[1] * (target * torch.log(out)) + weights[0] * ((1 - target) * torch.log(1 - out))
        else:
            loss = target * torch.log(out) + (1 - target) * torch.log(1 - out)
        return torch.neg(torch.mean(loss))

def shrink_box(x1, y1, w, h, shrink_factor):
    width = w
    height = h

    # calculating the width and height by shrinking
    new_width = int(width * shrink_factor)
    new_height = int(height * shrink_factor)

    # calculating the coordinates of center
    center_x = x1 + w / 2
    center_y = y1 + h / 2

    # Calculating the coordinates of the top left and bottom right corners of the shrunken box
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)

    # return the coordinates of the shrunken box
    return new_x1, new_y1, new_width, new_height

def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')

    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)
    log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')

    val_opt = copy.deepcopy(opt)
    val_opt.eval = True
    
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=val_opt)
    valid_dataset, _ = hierarchical_dataset(root=opt.valid_data, opt=val_opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
        
    """ model configuration """
    converter = TokenLabelConverter(opt)

    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    print("the network is {}".format(model))

    # data parallel for multi-GPU
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    
    model.train()
    
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'), strict=False)

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    mask_bceloss = WeightedBCELoss().to(device)
        
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    scheduler = None
    # optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=0.38328, rho=opt.rho, eps=opt.eps)

    if opt.scheduler:
        # scheduler = CosineAnnealingLR(optimizer, T_max=int(opt.num_iter))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000000)

    """ final options """
    # print(opt)
    with open(f'{opt.saved_path}/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        #print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    iteration = start_iter
            
    print("LR",scheduler.get_last_lr()[0])
        
    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        # iter_start_time = time.time()


        text=[]
        maskset=[]
        if not opt.is_mask_supervision:

            len_target, char_target = converter.char_encode(labels)
            char_preds,fpn_features = model(image)

            char_loss = criterion(char_preds.view(-1, char_preds.shape[-1]), char_target.contiguous().view(-1))
            cost = char_loss     
        else:
            for label in labels:
                bboxLabel,width,height=label.strip("\n").split("_")

                # processing box infor to adapt new size
                # generate mask and circulating to generate small mask
                width_ratio=int(width)/128
                height_ratio=int(height)/32
                bboxLabel=eval(bboxLabel)
                word=bboxLabel["word"]
                length = len(bboxLabel.keys())
                mask=np.zeros((32,128),dtype=np.uint8)
                for index,(boxori,char) in enumerate(list(bboxLabel.items())):
                    if index+1 == length:
                        break
                    box=copy.deepcopy(boxori)
                    # print("box",box)
                    box=eval(box)
                    box = np.array(box)
                    box[:,0]=(box[:,0]*(1/width_ratio)).astype(int)
                    box[:,1]=(box[:,1]*(1/height_ratio)).astype(int)
                    x = box[0][0]
                    y = box[0][1]
                    w = abs(box[1][0] - box[0][0])
                    h = abs(box[1][1] - box[0][1])
                    x, y, w, h=shrink_box(x, y, w, h, 0.5)
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                    
                text.append(word)
                maskset.append(mask)

            labels = text
            maskOriginal = [transforms.ToTensor()(m) for m in maskset]
            maskOriginal = torch.unsqueeze(torch.cat(maskOriginal,0),dim=1).to(device)

            len_target, char_target = converter.char_encode(labels)
            char_preds,fpn_features = model(image)
            mask_loss = mask_bceloss(fpn_features,maskOriginal,[1,2])

            char_loss = criterion(char_preds.view(-1, char_preds.shape[-1]), char_target.contiguous().view(-1))
            cost = char_loss + 0.5*mask_loss

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()
        loss_avg.add(cost)

        # validation part
        if utils.is_main_process() and ((iteration + 1) % opt.valInterval == 0 or iteration == 0): # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            print("LR",scheduler.get_last_lr()[0])
            with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, current_accuracys, char_preds, confidence_score, labels, infer_time, length_of_data, _ = validation(
                        model, criterion, valid_loader, converter, opt)
                    char_accuracy = current_accuracys[0]
                    bpe_accuracy = current_accuracys[1]
                    wp_accuracy = current_accuracys[2]
                    final_accuracy = current_accuracys[3]
                    cur_best = max(char_accuracy, bpe_accuracy, wp_accuracy, final_accuracy)
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] LR: {scheduler.get_last_lr()[0]:0.5f}, Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"char_accuracy":17s}: {char_accuracy:0.3f}, {"bpe_accuracy":17s}: {bpe_accuracy:0.3f}, {"wp_accuracy":17s}: {wp_accuracy:0.3f}, {"fused_accuracy":17s}: {final_accuracy:0.3f}'

                # keep best accuracy model (on valid dataset)
                if cur_best > best_accuracy:
                    best_accuracy = cur_best
                    torch.save(model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/best_accuracy.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], char_preds[:5], confidence_score[:5]):
                    if opt.Transformer:
                        pred = pred[:pred.find('[s]')]
                    elif 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                    if opt.sensitive and opt.data_filtering_off:
                        pred = pred.lower()
                        gt = gt.lower()
                        alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                        out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                        pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                        gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                # print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if utils.is_main_process() and (iteration + 1) % 5e+3 == 0:
            torch.save(
                model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
        if scheduler is not None:
            scheduler.step()

if __name__ == '__main__':

    opt = get_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'{opt.saved_path}/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    utils.init_distributed_mode(opt)
    
    """ Seed and GPU setting """
    
    seed = opt.manualSeed + utils.get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    train(opt)
 
