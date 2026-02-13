import numpy as np
import os
import glob
import torch
import argparse

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=160)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--train_eposide', type=int, help='meta batch size, namely task num', default=500)  #
    argparser.add_argument('--test_eposide', type=int, help='meta batch size, namely task num', default=600)  #
    argparser.add_argument('--k_shot', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--n_query', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=224)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)  # 4
    argparser.add_argument('--pre_lr', type=float, help='meta-level outer learning rate', default=0.0005)

    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    argparser.add_argument('--lr_LAT', type=float, help='the lr of LAT model',
                           default=0.001)
    argparser.add_argument('--update_LAT_lr', type=float, help='the lr after LAT to update meta model',
                           default=1e-3)
    argparser.add_argument('--device', type=int, help='the number of using cuda device', default=0)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=8)  # 5
    # argparser.add_argument('--inner_update_step', type=int, help='task-level inner update steps', default=3)  # 5
    argparser.add_argument('--LAT_update_step', type=int, help='task-level inner update steps', default=4)  # 5
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)  # 10
    argparser.add_argument('--loss_mode', type=str, help='the focal loss or entropy loss',
                           default='entropy')
    argparser.add_argument('--root_path', type=str, help='the root path of datasets',
                           default='./dataset/')
    argparser.add_argument('--test_dataset', type=str, help='the focal loss or entropy loss',
                           default='aid')
    argparser.add_argument('--pretrain_model_path', type=str, help='the path of train dataset',
                           default='./outputs/pretrain_LAT/new_result.pth')
    argparser.add_argument('--m', type=float, help='the coffient of moment',
                           default=0.998)
    argparser.add_argument('--update_m1', type=float, help='the coffient of moment',
                           default=0.5)
    argparser.add_argument('--update_m2', type=float, help='the coffient of moment',
                           default=0.1)
    argparser.add_argument('--update_m3', type=float, help='the coffient of moment',
                           default=0.9)
    argparser.add_argument('--sim_loss_alpha', type=float, help='the coffient of sim_loss',
                           default=1.5)
    argparser.add_argument('--grad_weight', type=float, help='the weight of sum grad',
                           default=0.001)
    argparser.add_argument('--sim_loss_mode', type=str, help='loss_mode',
                           default='ex')
    argparser.add_argument('--lamba2', type=float, default=0.15, help='lamba_intra')
    argparser.add_argument('--lamba1', type=float, default=1.0, help='lamba_cross')
    argparser.add_argument('--lamba3', type=float, default=0.8, help='lamba_LAT_loss_sim_inter')
    argparser.add_argument('--lamba4', type=float, default=0.8, help='lamba_LAT_loss_sim_recip')

    argparser.add_argument('--pretrain', type=str, help='the focal loss or entropy loss',
                           default='yes')
    argparser.add_argument('--change_lr', type=str, help='while change lr in training stage',
                           default='yes')
    argparser.add_argument('--backbone', type=str, help='the backbone',
                           default='resnet10_LAT_backbone')
    argparser.add_argument('--backbone_file', type=str, help='the backbone',
                           default='') #'grad_weight_0.05_dec'
    argparser.add_argument('--LAT_network', type=str, help='the backbone',
                           default='resnet10_LAT')
    argparser.add_argument('--per_ep_test_LAT', type=int, help='', default=3)  # 10
    argparser.add_argument('--per_step_LAT', type=int, help='', default=2)  # 10

    return argparser
