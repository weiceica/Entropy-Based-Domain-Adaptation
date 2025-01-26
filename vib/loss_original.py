import torch
import torch.nn as nn

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num)
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY - YX)
        return loss

def CORAL(source, target):
    d = source.size(1)
    source_cov = (source.t() @ source) / (source.size(0) - 1)
    target_cov = (target.t() @ target) / (target.size(0) - 1)
    loss = (source_cov - target_cov).pow(2).sum().sqrt()
    loss /= (4 * d * d)
    return loss
