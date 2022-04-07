import os
import cv2
import torch
import logging
import numpy as np
from utils.config import CONFIG
import torch.distributed as dist
import torch.nn.functional as F


def make_dir(target_dir):
    """
    Create dir if not exists
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def print_network(model, name):
    """
    Print out the network information
    """
    logger = logging.getLogger("Logger")
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()

    logger.info(model)
    logger.info(name)
    logger.info("Number of parameters: {}".format(num_params))


def update_lr(lr, optimizer):
    """
    update learning rates
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr(init_lr, step, iter_num):
    """
    Warm up learning rate
    """
    return step/iter_num*init_lr


def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict


def get_unknown_tensor(trimap):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    if CONFIG.model.trimap_channel == 3:
        weight = trimap[:, 1:2, :, :].float()
    else:
        weight = trimap.eq(1).float()
    return weight


def reduce_tensor_dict(tensor_dict, mode='mean'):
    """
    average tensor dict over different GPUs
    """
    for key, tensor in tensor_dict.items():
        if tensor is not None:
            tensor_dict[key] = reduce_tensor(tensor, mode)
    return tensor_dict


def reduce_tensor(tensor, mode='mean'):
    """
    average tensor over different GPUs
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'mean':
        rt /= CONFIG.world_size
    elif mode == 'sum':
        pass
    else:
        raise NotImplementedError("reduce mode can only be 'mean' or 'sum'")
    return rt


Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]


def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape
    pred = F.interpolate(pred, size=(640,640), mode='nearest')
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred < 1.0/255.0] = 0
    uncertain_area[pred > 1-1.0/255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1

    weight = np.array(weight, dtype=np.float)
    weight = torch.from_numpy(weight).cuda()

    weight = F.interpolate(weight, size=(H,W), mode='nearest')

    return weight