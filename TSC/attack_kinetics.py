import argparse
import os
#import torch
import numpy as np
import math
import attack_methods
from dataset.kinetics import get_dataset
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, OPT_PATH, get_cfg_custom
import subprocess

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N')
    parser.add_argument('--model', type=str, default='i3d_resnet101', help='i3d_resnet101 | slowfast_resnet101 | tpn_resnet101.')
    parser.add_argument('--attack_method', type=str, default='Temporal_Skip_Connection')
    parser.add_argument('--step', type=int, default=1, metavar='N',help='Multi-step or One-step.')
    parser.add_argument('--momentum', action='store_true', default=True, help='Use iterative momentum in MFFGSM.')

    # parameters in the paper
    parser.add_argument('--L', type=int, default=15, metavar='N')
    parser.add_argument('--kernel_mode', type=str, default='uniform',help='gaussian | linear | uniform')
    parser.add_argument('--top_k',type=int,default=4,help='the value of k in the paper')
    parser.add_argument('--mask_num',type=int,default=15,help= 'the value of p in the paper')

    args = parser.parse_args()
    

    args.adv_path = os.path.join(OPT_PATH, 'K400-{}-{}-kernel{}-{}-mask_num{}-top{}'.format(\
        args.model,args.step,args.L,args.kernel_mode,args.mask_num,args.top_k))

    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print (args)

    # loading cfg.
    cfg_path = CONFIG_PATHS[args.model]
    cfg = get_cfg_custom(cfg_path, args.batch_size)

    # loading dataset and model.
    name = cfg.CONFIG.MODEL.NAME.lower()   
    dataset_loader = get_dataset(cfg)
    model = get_model(cfg).cuda()

    # attack
    params = {'kernlen':(2*args.L-1), 
              'momentum':args.momentum,
              'kernel_mode':args.kernel_mode}
    
    attack_method = getattr(attack_methods,args.attack_method)(\
        model, mask_num = args.mask_num,params=params,steps=args.step,top_k = args.top_k)

    for step, data in enumerate(dataset_loader):
        if step %1 == 0:
            print ('Running {}, {}/{}'.format(args.attack_method, step+1, len(dataset_loader)))
        val_batch = data[0].cuda()
        val_label = data[1].cuda()
        adv_batches = attack_method(val_batch, val_label)
        val_batch = val_batch.detach()
        for ind,label in enumerate(val_label):
            ori = val_batch[ind].cpu().numpy()
            adv = adv_batches[ind].cpu().numpy()
            np.save(os.path.join(args.adv_path, '{}-adv'.format(label.item())), adv)
            np.save(os.path.join(args.adv_path, '{}-ori'.format(label.item())), ori)
    adv_path_basename = os.path.basename(args.adv_path)