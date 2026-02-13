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


    

def meta_learning(x, model_backbone, Siamese_backbone, model_header, optimizer, softmax, loss_fn, top1):
    if optimizer!='':
        optimizer.zero_grad()
    x_96 = torch.stack(x[2:8]).cuda()  # (6,way,shot+query,3,96,96)
    x_224 = torch.stack(x[8:]).cuda()  # (1,way,shot+query,3,224,224)
    support_set_anchor = x_224[0, :, :args.k_shot, :, :, :]  # (way,shot,3,224,224)
    query_set_anchor = x_224[0, :, args.k_shot:, :, :, :]  # (way,query,3,224,224)
    query_set_aug_96 = x_96[:, :, args.k_shot:, :, :, :]  # (6,way,query,3,96,96)
    temp_224 = torch.cat((support_set_anchor, query_set_anchor), 1)  # (way,shot+query,3,224,224)
    temp_224 = temp_224.contiguous().view(args.n_way * (args.k_shot + args.n_query), 3, 224,
                                          224)  # (way*(shot+query),3,224,224)
    temp_224, mid_output = model_backbone(temp_224,while_zip=True)  # (way*(shot+query),512) if LAT is in backbone, there are two outputs
    temp_224 = temp_224.view(args.n_way, args.k_shot + args.n_query, 512)  # (way,shot+query,512)
    support_set_anchor = temp_224[:, :args.k_shot, :]  # (way,shot,512)
    support_set_anchor = torch.mean(support_set_anchor, 1)  # (way, 512)
    query_set_anchor = temp_224[:, args.k_shot:, :]  # (way,query,512)
    query_set_anchor = query_set_anchor.contiguous().view(args.n_way * args.n_query, 512).unsqueeze(
        0)  # (1,way*query,512)

    query_set_aug_96 = query_set_aug_96.contiguous().view(6 * args.n_way * args.n_query, 3, 96,
                                                          96)  # (6*way*query,3,96,96)
    with torch.no_grad():
        query_set_aug_96, mid_output_s = Siamese_backbone(query_set_aug_96)  # (6*way*query,512)
    query_set_aug_96 = query_set_aug_96.view(6, args.n_way * args.n_query, 512)  # (6, 5*15, 512)
    query_set = torch.cat((query_set_anchor, query_set_aug_96), 0)  # (7, 5*15, 512)
    query_set = query_set.contiguous().view(7 * args.n_way * args.n_query, 512)

    pred_query_set = model_header(support_set_anchor, query_set)  # (7*5*15,5)

    pred_query_set = pred_query_set.contiguous().view(7, args.n_way * args.n_query,
                                                      args.n_way)  # (7,75,5)

    pred_query_set_anchor = pred_query_set[0]  # (75,5)
    pred_query_set_aug = pred_query_set[1:]  # (6,75,5)

    query_set_y = torch.from_numpy(np.repeat(range(args.n_way), args.n_query))
    query_set_y = Variable(query_set_y.cuda())
    ce_loss = loss_fn(pred_query_set_anchor, query_set_y)

    pred_query_set_anchor = softmax(pred_query_set_anchor)

    pred_query_set_aug = pred_query_set_aug.contiguous().view(6 * args.n_way * args.n_query, args.n_way)
    pred_query_set_aug = softmax(pred_query_set_aug)
    pred_query_set_anchor = torch.cat([pred_query_set_anchor for _ in range(6)], dim=0)
    self_image_loss = torch.mean(torch.sum(torch.log(pred_query_set_aug ** (-pred_query_set_anchor)), dim=1))

    pred_query_set_global = pred_query_set[0]  # (75,5)
    pred_query_set_global = pred_query_set_global.view(args.n_way, args.n_query, args.n_way)

    rand_id_global = np.random.permutation(args.n_query)
    pred_query_set_global = pred_query_set_global[:, rand_id_global[0], :]  # (way,way)
    pred_query_set_global = softmax(pred_query_set_global)  # (way,way)
    pred_query_set_global = pred_query_set_global.unsqueeze(0)  # (1,5,5)
    pred_query_set_global = pred_query_set_global.expand(6, args.n_way, args.n_way)  # (6,5,5)
    pred_query_set_global = pred_query_set_global.contiguous().view(6 * args.n_way,
                                                                    args.n_way)  # (6*way,way)

    rand_id_local_sample = np.random.permutation(args.n_query)
    pred_query_set_local = pred_query_set_aug.view(6, args.n_way, args.n_query, args.n_way)
    pred_query_set_local = pred_query_set_local[:, :, rand_id_local_sample[0], :]  # (6,way,way)
    pred_query_set_local = pred_query_set_local.contiguous().view(6 * args.n_way, args.n_way)  # (6*way,way)

    cross_image_loss = torch.mean(torch.sum(torch.log(pred_query_set_local ** (-pred_query_set_global)), dim=1))

    loss = ce_loss + self_image_loss * args.lamba1 + cross_image_loss * args.lamba2

    _, predicted = torch.max(pred_query_set[0].data, 1)
    correct = predicted.eq(query_set_y.data).cpu().sum()
    top1.update(correct.item() * 100 / (query_set_y.size(0) + 0.0), query_set_y.size(0))
    if optimizer == '':
        grad = torch.autograd.grad(loss, model_backbone.parameters())
    else:
        loss.backward()
        grad = ()
        for param in model_backbone.parameters():
            grad += (param.grad,)
        optimizer.step()
    return loss, grad, mid_output

def main():

    torch.manual_seed(2333)
    torch.cuda.manual_seed_all(2333)
    np.random.seed(2333)

    print(args)



    uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    uuid_str = str(uuid_str)

    device_to = torch.device('cuda:{}'.format(str(args.device)))
    
    dataset_list = os.listdir(args.root_path)
    
    # 过滤出目录（子文件夹）
    dataset_list = [dataset for dataset in dataset_list if os.path.isdir(os.path.join(args.root_path, dataset))]

    dataloader_list = []
    testloader_list = []
    for data in dataset_list:
        name = data
        train_path = os.path.join(args.root_path, name, 'train')
        class_list = os.listdir(train_path)
        # 过滤出目录（子文件夹）
        class_list = [class_name for class_name in class_list if os.path.isdir(os.path.join(train_path, class_name))]
        num_class = len(class_list)
        test_path = os.path.join(args.root_path, name, 'test')
        class_list = os.listdir(test_path)
        # 过滤出目录（子文件夹）
        class_list = [class_name for class_name in class_list if os.path.isdir(os.path.join(test_path, class_name))]
        test_num_class = len(class_list)
        data_train = train_dataset.Eposide_DataManager(data_path=train_path, num_class=num_class,
                                                           n_way=args.n_way, n_support=args.k_shot, n_query=args.n_query, n_eposide=args.train_eposide)
        train_loader = data_train.get_data_loader()
        if name == args.test_dataset:
            n_eposide = args.test_eposide
            datamgr = test_dataset.Eposide_DataManager(data_path=test_path,
                                                           num_class=test_num_class, image_size=args.imgsz,
                                                           n_way=args.n_way, n_support=args.k_shot,
                                                           n_query=args.n_query, n_eposide=n_eposide)
            novel_loader = datamgr.get_data_loader(aug=False)
        else:
            # n_eposide = 5
            n_eposide = args.test_eposide
            datamgr = train_dataset.Eposide_DataManager(data_path=test_path,
                                                           num_class=test_num_class,
                                                           n_way=args.n_way, n_support=args.k_shot,
                                                           n_query=args.n_query, n_eposide=n_eposide)
            novel_loader = datamgr.get_data_loader()

        if name == args.test_dataset:
            testloader_list.append((train_loader, novel_loader))
        else:
            dataloader_list.append((train_loader, novel_loader))

    model_backbone = Learner(args.backbone, args.n_way, args.k_shot, args.n_query)
    model_LAT = Learner(args.LAT_network, args.n_way, args.k_shot, args.n_query)
    model_header = ProtoNet.ProtoNet()

    tmp = filter(lambda x: x.requires_grad, model_backbone.parameters())

    if args.pretrain == 'yes':
        if args.backbone == 'resnet10_LAT_backbone':
            data_state = torch.load(args.pretrain_model_path)
            target_path = model_backbone.state_dict()
            state = OrderedDict()
            for key, value in zip(target_path.keys(), data_state['state'].values()):
                state[key] = value
            model_backbone.load_state_dict(state)
            model_LAT.load_state_dict(data_state['LAT_model'])
            del data_state, target_path, state

    Siamese_backbone = copy.deepcopy(model_backbone)
    Siamese_backbone = Siamese_backbone.cuda()
    meta_backbone = copy.deepcopy(model_backbone)
    meta_backbone = meta_backbone.cuda()
    Siamese_meta_backbone = copy.deepcopy(model_backbone)
    Siamese_meta_backbone = Siamese_meta_backbone.cuda()

    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model_backbone, model_header)
    print('Total trainable tensors:', num)

    if not os.path.exists('./outputs/CDML_meta/{}/{}/{}_way_{}_shot/logs/{}_{}_way_{}_shot/'.format( args.test_dataset,args.backbone, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot)):
        os.mkdir('./outputs/CDML_meta/{}/{}/{}_way_{}_shot/logs/{}_{}_way_{}_shot/'.format(args.test_dataset,args.backbone, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot))
    logger = get_logger('./outputs/CDML_meta/{}/{}/{}_way_{}_shot/logs/{}_{}_way_{}_shot/{}_{}_way_{}_shot.log'.format(args.test_dataset,args.backbone, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot))
    logger.info("{}\n\nstart training!\n{}".format(str(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())), str(args)))

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([{"params":model_backbone.parameters()}], lr=args.meta_lr)
    optimizer_scheduler = StepLR(optimizer, step_size=50,gamma=0.9)
    optimizer_meta = torch.optim.Adam([{"params":meta_backbone.parameters()}], lr=args.update_lr)
    optimizer_meta_scheduler = StepLR(optimizer_meta, step_size=50, gamma=0.85)
    optimizer_LAT = torch.optim.Adam([{"params": model_LAT.parameters()}], lr=args.lr_LAT)
    optimizer_LAT_scheduler = StepLR(optimizer_LAT, step_size=20, gamma=0.90)


    acc_val_max = 0.0
    acc_test_max = 0.0
    h_best = 0.0

    step_test = 0
    x_step = 0

    torch.cuda.empty_cache()
    train_list = dataloader_list[:-1]
    pesudo_list = dataloader_list[-1]

    # 初始测试结果
    accs_all_test = []


    for ep in range(args.epoch):
        model_backbone.train()
        top1 = utils.AverageMeter()
        total_loss = 0
        softmax = torch.nn.Softmax(dim=1)
        skip_i = 0
        # step_max = max(len(dataloader_list[0][0]), len(dataloader_list[1][0]), len(dataloader_list[2][0]), len(dataloader_list[3][0]))
        step_max = args.update_step
  
        for i in range(step_max):
            if i % (args.per_step_LAT) == 0:
                optim = ''
            else:
                optim = optimizer
            grad_list = []
            mid_output_list = []
            random.shuffle(dataloader_list)
            for dataloader in dataloader_list[:-1]:
                x = next(iter(dataloader[0]))
                loss, grad, mid_output = meta_learning(
                    x, model_backbone, Siamese_backbone, model_header, optim, softmax, loss_fn, top1
                )
                # if ep%2==0:
                grad_list.append(grad)
                mid_output_list.append(mid_output)
                    # for param, new_param in zip(Siamese_backbone.parameters(), fast_weights_S):
                    #     param.data.copy_(new_param)
                if i % (args.per_step_LAT) != 0:
                    with torch.no_grad():
                        for param_q, param_k in zip(model_backbone.parameters(), Siamese_backbone.parameters()):
                            param_k.data = param_k.data * args.update_m1 + param_q.data * (1. - args.update_m1)   # 表示动量更新的系数
            if i%(args.per_step_LAT)==0:
                fast_weights = LAT(mid_output_list, model_backbone, model_LAT, args.grad_weight, grad_list, args.meta_lr)
                with torch.no_grad():
                    for param, new_param in zip(model_backbone.parameters(), fast_weights):
                        param.data.copy_(new_param)
                    for param_q, param_k in zip(model_backbone.parameters(), Siamese_backbone.parameters()):
                        param_k.data = param_k.data * args.update_m1 + param_q.data * (1. - args.update_m1)  # 表示动量更新的系数

            if ep%args.per_ep_test_LAT==0:
                grad_list = []
                mid_output_list = []
                for dataloader_t in dataloader_list[:-1]:
                    x_t = next(iter(dataloader_t[0]))
                    loss, grad, mid_output = meta_learning(
                        x_t, model_backbone, Siamese_backbone, model_header, '', softmax, loss_fn, top1
                    )
                    contain_nan = False
                    for i_g, tensor in enumerate(grad):
                        if torch.isnan(tensor).any():
                            contain_nan = True
                            break
                    if contain_nan:
                        continue
                    grad_list.append(grad)
                    mid_output_list.append(mid_output)
                for dataloader_t in dataloader_list[:-1]:
                    x_t = next(iter(dataloader_t[1]))[0]
                    # x_data = x_t[0]
                    loss_LAT,grad_LAT,fast_weights = test_LAT(x_t, mid_output_list, model_backbone, model_header,
                                                                        grad_list, args.meta_lr/len(grad_list), model_LAT, softmax,
                                                                        loss_fn,top1, args.grad_weight)
                    with torch.no_grad():
                        for param, g in zip(model_LAT.parameters(), grad_LAT):
                            param.data.sub_(args.lr_LAT * g)

            with torch.no_grad():
                for param_q, param_k in zip(model_backbone.parameters(), meta_backbone.parameters()):
                    param_k.data = param_k.data * args.update_m2 + param_q.data * (1. - args.update_m2)
                for param_q, param_k in zip(Siamese_backbone.parameters(), Siamese_meta_backbone.parameters()):
                    param_k.data = param_k.data * args.update_m2 + param_q.data * (1. - args.update_m2)
        if ep%args.per_ep_test_LAT==0:
            optimizer_scheduler.step()
        for test_i in range(args.update_step_test):
            x = next(iter(dataloader_list[-1][1]))
            grad_list = []
            mid_output_list = []
            loss, grad, mid_output = meta_learning(
                x, meta_backbone, Siamese_meta_backbone, model_header, optimizer_meta, softmax, loss_fn, top1
            )
            total_loss = total_loss + loss.item()
            avg_loss = total_loss / float(test_i + 1 - skip_i)
            avg_acc = top1.avg

            with torch.no_grad():
                for param_q, param_k in zip(meta_backbone.parameters(), Siamese_meta_backbone.parameters()):
                    param_k.data = param_k.data * args.update_m1 + param_q.data * (1. - args.update_m1)  # 表示动量更新的系数

                for param_q, param_k in zip(meta_backbone.parameters(), model_backbone.parameters()):
                    param_k.data = param_q.data
                for param_q, param_k in zip(Siamese_meta_backbone.parameters(), Siamese_backbone.parameters()):
                    param_k.data = param_q.data


        optimizer_meta_scheduler.step()



        logger.info('train: {:d}, current epoch train loss: {:.3f}, current epoch train acc: {:.3f}'.format(ep+1, avg_loss, avg_acc))

        train_result = {
            'epoch': ep + 1,
            'state_model': meta_backbone.state_dict(),
            'state_Siamese_model': Siamese_meta_backbone.state_dict()
        }
        torch.save(train_result,'./outputs/CDML_meta/{}/{}/{}_way_{}_shot/logs/{}_{}_way_{}_shot/new_result.pth'.format(args.test_dataset,args.backbone, args.n_way, args.k_shot, uuid_str, args.n_way, args.k_shot))

        if (ep+1) % 20 == 0:
            meta_backbone.eval()
            logger.info("after train: ")
            test(testloader_list[0][1], meta_backbone, args, logger)
            # logger.info('test acc:\t acc={:.3f}+-{:.3f}!!!!  best_acc={:.3f}+-{:.3f}!!!'.format(float(accs), float(h), float(acc_test_max), float(h_best)))
    del model_backbone, Siamese_backbone
    data_state = torch.load(args.pretrain_model_path)
    target_path = meta_backbone.state_dict()
    state = OrderedDict()
    for key, value in zip(target_path.keys(), data_state['state'].values()):
        state[key] = value
    # state = data_state['state']
    meta_backbone.load_state_dict(state)
    meta_backbone.cuda()
    meta_backbone.eval()
    logger.info("before train: ")
    test(testloader_list[0][1], meta_backbone, args, logger)



if __name__ == '__main__':


    argparser = parse_args()

    args = argparser.parse_args()
    main()