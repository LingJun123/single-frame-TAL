from __future__ import print_function
import argparse
import os
import torch
from model import Model
from video_dataset import Dataset
from test import test
from stage2 import train
from tensorboard_logger import Logger
import options
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.optim as optim

if __name__ == '__main__':

    args = options.parser.parse_args()
    torch.manual_seed(args.seed)            # args.seed = 1
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    
    t_max = 750
    t_max_ctc = 2800
    if args.activity_net:
        t_max = 200
        t_max_ctc = 400
    dataset = Dataset(args)
    os.system('mkdir -p ./ckpt/')
    os.system('mkdir -p ./logs/' + args.model_name)
    logger = Logger('./logs/' + args.model_name)
    model = Model(dataset.feature_size, dataset.num_class)
    args.pretrained_ckpt = './ckpt/pointNet5000.pkl'
    if args.eval_only and args.pretrained_ckpt is None:
        print('***************************')
        print('Pretrained Model NOT Loaded')
        print('Evaluating on Random Model')
        print('***************************')

    if args.pretrained_ckpt is not None:
       print('model load!')
       model.load_state_dict(torch.load(args.pretrained_ckpt))
    print('begin')
    best_acc = 0
    # baseline = [59.1,53.5,44.2,34.1,26.6,8.1]
    best05 = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    args.max_iter = 12001
    for itr in range(5001,args.max_iter):
        dataset.t_max = t_max
        if itr % 2:      # itr 为偶数，且不为零
            dataset.t_max = -1
        if not args.eval_only:      # eval_only = False
            train(itr, dataset, args, model, optimizer, logger,device)
        N = 50
        if itr % N == 0 and (not itr == 0 or args.eval_only):     # 迭代500次或其倍数

            res = test(itr, dataset, args, model, logger, device)   # 测试
            acc = res[6]
            acc05 = res[4]
            torch.save(model.state_dict(), './ckpt/' + 'currentPointNet' + '.pkl')
            if acc05 > best05 and not args.eval_only:
                best05 = acc05
                torch.save(model.state_dict(), './ckpt/' + 'pointNet05' + '.pkl')
            if acc > best_acc and not args.eval_only:
                best_acc = acc
                torch.save(model.state_dict(), './ckpt/' + 'pointNet' + '.pkl')
        if args.eval_only:
            print('Done Eval!')
            break
