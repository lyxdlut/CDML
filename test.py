import copy
import time
import  torch, os
import  numpy as np
import scipy.stats as stats
import  scipy.stats
from    torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn as nn
from    torch.optim import lr_scheduler
import  random, sys, pickle
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d, AvgPool2d
from Option import parse_args
import utils
import logging
from torch.autograd import Variable
import  ProtoNet
from    learner import Learner
from    copy import deepcopy
import train_dataset
import test_dataset
from sklearn.linear_model import LogisticRegression
from torchvision import models
from torch.optim.lr_scheduler import StepLR
from LAT_utils import LAT, test_LAT

import argparse
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG,  1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def test(novel_loader, model, args, logger):
    iter_num = len(novel_loader)
    acc_all_LR = []
    with torch.no_grad():
        for i, (x,_) in enumerate(novel_loader):
            x_query = x[:, args.k_shot:,:,:,:].contiguous().view(args.n_way*args.n_query, *x.size()[2:]).cuda()
            x_support = x[:,:args.k_shot,:,:,:].contiguous().view(args.n_way*args.k_shot, *x.size()[2:]).cuda() # (25, 3, 224, 224)
            out_support, mid_output = model(x_support) # (way*shot,512)
            out_query, mid_output_q = model(x_query) # (way*query,512)
            # del mid_output, mid_output_q

            beta = 0.5
            out_support = torch.pow(out_support, beta)
            out_query = torch.pow(out_query, beta)

            _, c = out_support.size()

            out_support_LR_with_GC = out_support.cpu().numpy()
            out_query_LR_with_GC = out_query.cpu().numpy()
            y = np.tile(range(args.n_way), args.k_shot)
            y.sort()
            classifier = LogisticRegression(max_iter=1000).fit(X=out_support_LR_with_GC, y=y)
            pred = classifier.predict(out_query_LR_with_GC)
            gt = np.tile(range(args.n_way), args.n_query)
            gt.sort()
            acc_LG = np.mean(pred == gt)*100.
            acc_all_LR.append(acc_LG)
    acc_all  = np.asarray(acc_all_LR)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    logger.info('test acc : %4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))





def main(args):

    torch.manual_seed(2333)
    torch.cuda.manual_seed_all(2333)
    np.random.seed(2333)

    print(args)



    uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    uuid_str = str(uuid_str)


    device_to = torch.device('cuda:{}'.format(str(args.device)))


    test_path = os.path.join(args.root_path, args.test_dataset, 'test')
    class_list = os.listdir(test_path)
    class_list = [class_name for class_name in class_list if os.path.isdir(os.path.join(test_path, class_name))]
    test_num_class = len(class_list)
    n_eposide = args.test_eposide
    datamgr = LDP_test_dataset.Eposide_DataManager(data_path=test_path,
                                                    num_class=test_num_class, image_size=args.imgsz,
                                                    n_way=args.n_way, n_support=args.k_shot,
                                                    n_query=args.n_query, n_eposide=n_eposide)
    novel_loader = datamgr.get_data_loader(aug=False)


    model_backbone = Learner(args.backbone, args.n_way, args.k_shot, args.n_query)
    tmp = filter(lambda x: x.requires_grad, model_backbone.parameters())


    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model_backbone)
    print('Total trainable tensors:', num)

    if not os.path.exists('./outputs/CDML_meta/{}/{}/{}_way_{}_shot/logs/{}_{}_way_{}_shot/'.format( args.test_dataset,args.backbone, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot)):
        os.mkdir('./outputs/CDML_meta/{}/{}/{}_way_{}_shot/logs/{}_{}_way_{}_shot/'.format(args.test_dataset,args.backbone, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot))
    logger = get_logger('./outputs/CDML_meta/{}/{}/{}_way_{}_shot/logs/{}_{}_way_{}_shot/{}_{}_way_{}_shot.log'.format(args.test_dataset,args.backbone, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot))




    torch.cuda.empty_cache()

    # 初始测试结果
    accs_all_test = []

    test_state = torch.load(args.weight_file)['state_model']
    model_backbone.load_state_dict(test_state)
    model_backbone.cuda()
    model_backbone.eval()
    logger.info("test result: ")
    test(novel_loader, model_backbone, args, logger)




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--test_eposide', type=int, help='meta batch size, namely task num', default=600)  #
    argparser.add_argument('--k_shot', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--n_query', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=224)
    
    argparser.add_argument('--device', type=int, help='the number of using cuda device', default=0)
    argparser.add_argument('--root_path', type=str, help='the root path of datasets',
                           default='./dataset/')
    argparser.add_argument('--test_dataset', type=str, help='the focal loss or entropy loss',
                           default='IR')
    argparser.add_argument('--backbone', type=str, help='the backbone',
                           default='resnet10_LDP_LAT_backbone')
    argparser.add_argument('--LAT_network', type=str, help='the backbone',
                           default='resnet10_LAT')

    argparser.add_argument('--weight_file', type=str, help='the backbone',
                           default='')
    args = argparser.parse_args()
    main(args)