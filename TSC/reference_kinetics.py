import os
import time
import numpy as np
import pandas as pd
import json

import torch
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, get_cfg_custom, AverageMeter, OPT_PATH
import argparse
import math

#torch.backends.cudnn.enabled = False

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--adv_path', type=str, default='', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for reference (default: 16)')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, args.adv_path)
    return args

def accuracy(output, target):
    batch_size = target.size(0)                                     # target.shape (16,)

    _, pred = output.topk(1, 1, True, True)                         # 用于返回top-1的索引，_为value,pred为index  True，True 表示返回最大元素，进行大到小排序
    pred = pred.t()                                                 # 1,batch_size   进行转置
    correct = pred.eq(target.view(1, -1).expand_as(pred))           # eq:对比pred和label是否相同(对比top1索引是否等于label，第几个就是第几类)，获得bool (1,16)

    correct_k = correct[:1].view(-1).float().sum(0)                 # [1,16]实际是一个二维数组，所以用view变一维，方便sum，float让bool值变为1.0或0.0
    return correct_k.mul_(100.0 / batch_size), torch.squeeze(pred)  # correct_k为预测正确的个数，/batch_size*100%得到准确率，torch.squeeze(pred)为batch中的top-1索引

def generate_batch(batch_files):
    batches = []
    labels = []
    for file in batch_files:
        batches.append(torch.from_numpy(np.load(os.path.join(args.adv_path, file))).cuda())
        labels.append(int(file.split('-')[0]))
    labels = np.array(labels).astype(np.int32)          # list  -> numpy
    labels = torch.from_numpy(labels)                   # numpy -> Tensor
    return torch.stack(batches), labels                 # torch.stack(batches).shape:(16,3,32,224,224) labels:(16,)

def reference(model, files_batch):
    data_time = AverageMeter()                                  # 计时
    top1 = AverageMeter()                                       # 记录top-1的平均准确率 
    batch_time = AverageMeter()                                

    predictions = []
    labels = []

    end = time.time()
    with torch.no_grad():
        for step, batch in enumerate(files_batch):
            data_time.update(time.time() - end)
            val_batch, val_label = generate_batch(batch)

            val_batch = val_batch.cuda()
            val_label = val_label.cuda()

            batch_size = val_label.size(0)
            outputs = model(val_batch)

            prec1a, preds = accuracy(outputs.data, val_label) # 无目标攻击 返回当前batch的准确率和top-1 index

            predictions += list(preds.cpu().numpy())          # k400，类别正好为0-399和index重合，predictions即模型预测结果
            labels += list(val_label.cpu().numpy())           # 真实标签

            top1.update(prec1a.item(), val_batch.size(0))     # 获取top-1准确率的均值
            batch_time.update(time.time() - end)
            end = time.time()

            if step % 5 == 0:
                print('----validation----')
                print_string = 'Process: [{0}/{1}]'.format(step + 1, len(files_batch))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'top-1 accuracy: {top1_acc:.2f}%'.format(top1_acc = top1.avg)
                print (print_string)
    return predictions, labels, top1.avg

if __name__ == '__main__':
    global args
    args = arg_parse()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # loading adversarial examples.
    files = os.listdir(args.adv_path)                                            # 对抗样本路径
    files = [i for i in files if 'adv' in i]

    batch_times = math.ceil(len(files) / args.batch_size)                        # 400/16 
    files_batch = []
    for i in range(batch_times):
        batch = files[i*args.batch_size: min((i+1)*args.batch_size, len(files))] # 一个batch数量的文件
        files_batch.append(batch)                                                # 每个batch批量放入files_batch

    model_val_acc = {}
    info_df = pd.DataFrame()                                                     # 生成一个类似表格的数据结构
    info_df['gt_label'] = [i for i in range(400)]                                # 生成标签，因为k400共400个类
    for model_name in CONFIG_PATHS.keys():
        print ('Model-{}:'.format(model_name))
        cfg_path = CONFIG_PATHS[model_name]
        cfg = get_cfg_custom(cfg_path)
        model = get_model(cfg).cuda()
        model.eval()
        
        preds, labels, top1_avg = reference(model, files_batch)                  # 输入一个batch和模型 返回模型预测结果，标签，top-1的准确率

        predd = np.zeros_like(preds)
        inds = np.argsort(labels)                                                # 返回labels从小到大的索引，比如：[3,1,2] ->[1,2,0]
        for i,ind in enumerate(inds):
            predd[ind] = preds[i]

        info_df['{}-pre'.format(model_name)] = predd
        model_val_acc[model_name] = 100-top1_avg
        del model
        torch.cuda.empty_cache()                                                 # 释放由 CUDA 分配的但当前未由分配器管理的缓存内存

    info_df.to_csv(os.path.join(args.adv_path, 'results_all_models_prediction.csv'), index=False)
    # with open(os.path.join(args.adv_path, 'top1_acc_all_models.json'), 'w') as opt:
    with open(os.path.join(args.adv_path, 'ASR.json'), 'w') as opt:
        json.dump(model_val_acc, opt)
    # delete the generated npy files.
    command = os.path.join(args.adv_path, '*.npy')
    os.system('rm {}'.format(command))
