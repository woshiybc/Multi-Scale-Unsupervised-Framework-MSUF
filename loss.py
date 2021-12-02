import torch
import numpy as np
import CFOG as des
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


def ComputeLoss(reference, sensed_tran, sensed, reference_inv_tran):
    loss_1 = CFOG_SSD(reference, sensed_tran)
    loss_2 = CFOG_SSD(sensed, reference_inv_tran)
    loss = loss_1 + loss_2
    return loss


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def CFOG_NCC(i, j):  # sar_flow, optical
    x = torch.ge(i.squeeze(0).squeeze(0), 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.ge(j.squeeze(0).squeeze(0), 1)
    y = torch.tensor(y, dtype=torch.float32)
    z = torch.mul(x, y)
    num = z[z.ge(1)].size()[0]
    i = torch.mul(i, z)
    j = torch.mul(j, z)
    CFOG_sar = torch.mul(des.CFOG(i), z)
    CFOG_optical = torch.mul(des.CFOG(j), z)
    # loss = gncc_loss(MIND_sar, MIND_optical)
    loss = gncc_loss(CFOG_sar, CFOG_optical)*512*512/num
    return loss


def gncc_loss(I, J, eps=1e-5):
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I*J
    I_ave, J_ave = I.mean(), J.mean()
    I2_ave, J2_ave = I2.mean(), J2.mean()
    IJ_ave = IJ.mean()
    cross = IJ_ave - I_ave * J_ave
    I_var = I2_ave - I_ave.pow(2)
    J_var = J2_ave - J_ave.pow(2)
    cc = cross / (I_var.sqrt() * J_var.sqrt() + eps)  # 1e-5
    return -1.0 * cc + 1


def CFOG_SSD(i, j):
    x = torch.ge(i.squeeze(0).squeeze(0), 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.ge(j.squeeze(0).squeeze(0), 1)
    y = torch.tensor(y, dtype=torch.float32)
    z = torch.mul(x, y)
    num = z[z.ge(1)].size()[0]
    i = torch.mul(i, z)
    j = torch.mul(j, z)
    CFOG_sar = torch.mul(des.CFOG(i), z)
    CFOG_optical = torch.mul(des.CFOG(j), z)
    SSD_loss = nn.MSELoss(reduction='sum')
    loss = SSD_loss(CFOG_sar, CFOG_optical)/num
    return loss


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)


def lncc_loss(i, j, win=[9, 9], eps=1e-5):
    I = i
    J = j
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I*J
    filters = Variable(torch.ones(1, 1, win[0], win[1])).cuda()
    padding = (win[0]//2, win[1]//2)
    I_sum = F.conv2d(I, filters, stride=1, padding=padding)
    J_sum = F.conv2d(J, filters, stride=1, padding=padding)
    I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
    J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
    IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)
    win_size = win[0] * win[1]
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    cc = cross * cross / (I_var * J_var + eps)
    lcc = -1.0 * torch.mean(cc) + 1
    return lcc
