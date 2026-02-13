import  torch, os
import torch.nn.functional as f
from torch.autograd import Variable
from Option import parse_args
import  numpy as np
def computer_norm(tensor_tuple):
    norm_tuple = tuple(torch.norm(tensor) for tensor in tensor_tuple)
    return norm_tuple
def calculate_cosine_similarity(tensor1, tensor2):
    # 将张量展平为一维向量
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)

    # 计算余弦相似度
    return f.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0))

def calculate_loss(x, alpha=1.0, sim_loss_mode="ex"):
    if sim_loss_mode == 'ex':
        return torch.where(x<=0,alpha*(torch.exp(-x)-1.0),0.0*x)
    elif sim_loss_mode == 'x2':
        return torch.where(x<=0,alpha * x * x,0.0*x)

def LAT(mid_output_list, model_backbone,model_LAT,grad_weight, grad_list, lr):
    argparser = parse_args()
    args = argparser.parse_args()
    grad_tran_list = []
    fast_weights = list(model_backbone.parameters())
    # fast_weights_S = list(Siamese_backbone.parameters())
    for i in range(len(mid_output_list)):
        # mid_output = mid_output_list[i]
        out_weight, out_bias = model_LAT(mid_output_list[i])
        # mean_output = mid_output.mean(dim=[2,3])
        # var_output = mid_output.var(dim=[2,3])
        # cat_output = torch.cat((mean_output, var_output), dim=1)
        norm_weight = computer_norm(out_weight)
        norm_bias = computer_norm(out_bias)
        norm_grad = computer_norm(grad_list[i])
        out_weight = tuple(a / b for a, b in zip(out_weight, norm_weight))
        # out_bias = tuple(a * (c / b) * 0.01 for a, b, c in zip(out_bias, norm_bias, norm_grad))

        grad_tran = tuple(b * g + w for w, b, g in zip(out_weight, out_bias, grad_list[i]))
        grad_tran = tuple(grad_weight * g2 + (1-grad_weight)*g1 for g1, g2 in zip(grad_list[i], grad_tran))
        grad_tran_list.append(grad_tran)

        fast_weights = list(map(lambda p: p[1] - lr * p[0], zip(grad_tran, fast_weights)))
        # fast_weights_S = [t2 * args.update_m1 + t1 * (1 - args.update_m1) for t1, t2 in
        #                   zip(fast_weights, fast_weights_S)]
    return fast_weights



def test_LAT(x, mid_output_list, model_backbone, model_header, grad_list,lr ,model_LAT, softmax, loss_fn,top1, grad_weight):
    # if optimizer!='':
    #     optimizer.zero_grad()
    argparser = parse_args()
    args = argparser.parse_args()
    grad_tran_list = []
    fast_weights = list(model_backbone.parameters())
    loss_sim_recip = torch.tensor([0.0]).to('cuda')
    for i in range(len(mid_output_list)):
        # mid_output = mid_output_list[i]
        out_weight, out_bias = model_LAT(mid_output_list[i])

        norm_weight = computer_norm(out_weight)
        norm_bias = computer_norm(out_bias)
        norm_grad = computer_norm(grad_list[i])
        out_weight = tuple(a / b for a, b in zip(out_weight, norm_weight))
        # out_bias = tuple(a * (c / b) * 0.0001 for a, b, c in zip(out_bias, norm_bias, norm_grad))

        grad_tran = tuple(b * g + w for w, b, g in zip(out_weight, out_bias, grad_list[i]))
        # norm_grad_tran = computer_norm(grad_tran)
        # grad_tran = tuple(a*b/c for a,b,c in zip(grad_tran, norm_grad,norm_grad_tran))

        grad_tran_view1 = ()
        grad_tran_view2 = ()
        for j in range(len(grad_tran)):
            # grad_tran_list[i][j] = grad_tran_list[i][j].view(-1)
            grad_tmp1 = grad_tran[j]
            grad_tmp2 = grad_list[i][j]
            grad_tran_view1 += (grad_tmp1.view(-1),)
            grad_tran_view2 += (grad_tmp2.view(-1),)
        grad_tran_view1 = torch.cat(grad_tran_view1)
        grad_tran_view2 = torch.cat(grad_tran_view2)
        sim = calculate_cosine_similarity(grad_tran_view1, grad_tran_view2)
        # similarity.append((i,j,sim))
        loss_sim_recip += calculate_loss(sim, args.sim_loss_alpha, args.sim_loss_mode)
        grad_tran = tuple(g1 + grad_weight * g2 for g1, g2 in zip(grad_list[i], grad_tran))
        grad_tran_list.append(grad_tran)

        fast_weights = list(map(lambda p: p[1] - lr * p[0], zip(grad_tran, fast_weights)))


    for i in range(len(grad_tran_list)):
        grad_tran_view = ()
        for j in range(len(grad_tran_list[i])):
            # grad_tran_list[i][j] = grad_tran_list[i][j].view(-1)
            grad_tran_view += (grad_tran_list[i][j].view(-1),)
        grad_tran_view =torch.cat(grad_tran_view)
        grad_tran_list[i] = grad_tran_view

    loss_sim_inter = torch.tensor([0.0]).to('cuda')

    for i in range(len(grad_tran_list)):
        for j in range(i+1,len(grad_tran_list)):
            sim = calculate_cosine_similarity(grad_tran_list[i], grad_tran_list[j])
            loss_sim_inter = loss_sim_inter + calculate_loss(sim, args.sim_loss_alpha, args.sim_loss_mode)
    loss_sim_inter = loss_sim_inter.squeeze()
    loss_sim_recip = loss_sim_recip.squeeze()


    x_query = x[:, args.k_shot:, :, :, :].contiguous().view(args.n_way * args.n_query, *x.size()[2:]).cuda()
    x_support = x[:, :args.k_shot, :, :, :].contiguous().view(args.n_way * args.k_shot,
                                                              *x.size()[2:]).cuda()  # (25, 3, 224, 224)
    out_support, _ = model_backbone(x_support, fast_weights)  # (way*shot,512)
    out_query, _ = model_backbone(x_query, fast_weights)
    beta = 0.5
    out_support_pow = torch.pow(out_support, beta)
    out_query_pow = torch.pow(out_query, beta)
    _, c = out_support.size()

    pred_query_set = model_header(out_support, out_query)
    query_set_y = torch.from_numpy(np.repeat(range(args.n_way), args.n_query))
    query_set_y = Variable(query_set_y.cuda())
    ce_loss = loss_fn(pred_query_set, query_set_y)


    loss_LAT = ce_loss + loss_sim_inter * args.lamba3 +loss_sim_recip * args.lamba4


    grad_LAT = torch.autograd.grad(loss_LAT, model_LAT.parameters())

    return loss_LAT,grad_LAT, fast_weights