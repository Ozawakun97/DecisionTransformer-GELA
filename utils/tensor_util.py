import torch


def fill_mask(src_tensor,mask,device):
    dimension = src_tensor.shape[-1]
    fill_tensor = torch.full((dimension,), float('-inf')).to(device)
    fill_tensor[37] = float('inf')
    src_tensor[~mask] = fill_tensor
    return src_tensor