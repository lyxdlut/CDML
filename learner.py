import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from config import Config



class Learner(nn.Module):
    """

    """

    def __init__(self, backbone,  class_num, k_shot, n_query):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        backbone: resnet18_fc or resnet18 or relationNet
        """


        super(Learner, self).__init__()
        config_data = Config(backbone, class_num, k_shot, n_query)
        self.backbone = backbone
        self.bone_config, self.header_config = config_data.return_config()
        if 'fc' in backbone and 'resnet' in backbone:
            self.config = self.bone_config + self.header_config
        elif 'fc' not in backbone and 'resnet' in backbone:
            self.config = self.bone_config
        elif 'resnet' not in backbone:
            self.config = self.header_config

        # this dict contains all tensors needed to be optimized
        device = torch.device('cuda')
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name in ['conv2d', 'conv2d_in', 'conv_down', 'conv_down_start']:
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]).to(device))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                # self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]).to(device))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                # self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name in ['linear', 'linear_gnn', 'linear_re', 'linear_b', 'linear_w']:
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param).to(device))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]).to(device)))


            elif name in ['bn', 'bn_out', 'bn_down']:
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]).to(device))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]).to(device)))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]).to(device), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]).to(device), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d', 'max_pool2d_r',
                          'flatten', 'output_layer' ,'relu_out','flatten_w','mean', 'reshape','reshape_w' ,'leakyrelu', 'relu_r', 'res_add', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
                continue

            if name is 'conv2d_in':
                tmp = 'conv2d_in:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
                continue

            if name is 'conv_down':
                tmp = 'conv_down:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
                continue

            if name is 'conv_down_start':
                tmp = 'conv_down:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
                continue

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
                continue

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'
                continue

            elif name is 'linear_re':
                tmp = 'linear_re:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'
                continue

            elif name is 'linear_w':
                tmp = 'linear_w:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'
                continue

            elif name is 'linear_b':
                tmp = 'linear_b:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'
                continue

            elif name is 'linear_gnn':
                tmp = 'linear_gnn:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'
                continue

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'
                continue

            elif name is 'res_add':
                tmp = 'res_add:(%f)' % (param[0])
                info += tmp + '\n'
                continue


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
                continue
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
                continue
            elif name is 'max_pool2d_r':
                tmp = 'max_pool2d_r:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
                continue
            elif name in ['flatten', 'output_layer','flatten_w', 'reshape_w','tanh', 'relu','relu_out', 'mean', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'bn_out', 'bn_down', 'relu_r', 'relu_down']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
                continue
            else:
                raise NotImplementedError

        return info


    def forward(self, x,  vars=None, bn_training=True, while_zip=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars
        mid_output = None
        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name in ['conv2d', 'conv2d_in']:
                w = vars[idx]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, None, stride=param[4], padding=param[5])
                idx += 1
                # print(name, param, '\tout:', x.shape)
            elif name is 'conv_down':
                w = vars[idx]
                residual = F.conv2d(residual, w, None, stride=param[4], padding=param[5])
                idx += 1
            elif name is 'conv_down_start':
                w = vars[idx]
                residual = F.conv2d(x, w, None, stride=param[4], padding=param[5])
                idx += 1
            elif name is 'convt2d':
                w = vars[idx]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, None, stride=param[4], padding=param[5])
                idx += 1
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            # elif name is 'linear_gnn':
            #     w, b = vars[idx], vars[idx + 1]
            #     neighbor_features = torch.index_select(x, 0, edge[1])
            #     x = F.linear()
            elif name is 'linear_re':
                w, b = vars[idx], vars[idx + 1]
                output = F.linear(x, w, b)
                idx += 2

            elif name is 'linear_w':
                w, b = vars[idx], vars[idx + 1]
                output = F.linear(x, w, b)
                output = output.squeeze()
                idx += 2
                out_weight += (output,)

            elif name is 'linear_b':
                w, b = vars[idx], vars[idx + 1]
                bias = F.linear(x, w, b)
                idx += 2
                out_bias = tuple(bias[:, [i]].view(1) for i in range(w.shape[0]))

            elif name in ['bn', 'bn_out']:
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'bn_down':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                residual = F.batch_norm(residual, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'res_add':
                x += residual
            elif name is 'flatten' :
                # print(x.shape)
                x = x.contiguous().view(x.size(0), -1)
            elif name is 'flatten_w' :
                # print(x.shape)
                # x = x.view(x.size(0), -1)
                x = x.contiguous().view(1, -1)
                out_weight = ()
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                if -1 not in param:
                    x = x.contiguous().view(x.size(0), *param)
                else:
                    x = x.contiguous().view(-1, x.size(2), x.size(3))
            elif name is 'reshape_w':
                # [b, 8] => [b, 2, 2, 2]
                output = output.contiguous().view(*param)
                out_weight += (output,)

            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'relu_out':
                x = F.relu(x, inplace=param[0])
                out_weight = x
            elif name is 'relu_r':
                x = F.relu(x, inplace=param[0])
                residual = x
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'max_pool2d_r':
                x = F.max_pool2d(x, param[0], param[1], param[2])
                residual = x
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name is 'mean':
                x = torch.mean(x, [2,3])
            elif name is 'output_layer':
                n_class = param[0]
                n_shot = param[1]
                k_query = param[2]
                if while_zip == True:
                    mid_output = x
                    mid_output = mid_output.contiguous().view(n_class, n_shot+k_query, x.size(1), x.size(2), x.size(3))
                    mean_output = mid_output[:,:n_shot,:,:,:].mean(dim=1)
                    remain_output = mid_output[:,n_shot:,:,:,:]
                    mid_output = torch.cat((mean_output.unsqueeze(1),remain_output),dim=1)
                    mid_output = mid_output.contiguous().view(-1,mid_output.size(2),mid_output.size(3),mid_output.size(4))
                    del mean_output,remain_output
                else:
                    mid_output = x



            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        if 'LAT' not in self.backbone:
            return x
        elif 'resnet10_LAT' in self.backbone:
            return out_weight, out_bias
        elif 'LAT' in self.backbone and 'backbone' in self.backbone:
            return x, mid_output
    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
