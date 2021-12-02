from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SE(nn.Module):
    def __init__(self, in_channel, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1, stride=1, padding=0)
        self.excitation = nn.Conv2d(in_channel//ratio, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.softmax(out)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SEBolck(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p, ratio=16):
        super(SEBolck, self).__init__()
        self.conv_block = ConvBlock(in_channel, out_channel, k, s, p)
        self.se = SE(out_channel, ratio)

    def forward(self, x):
        out = self.conv_block(x)
        se = self.se(out)
        out = out * se
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, strides, padding):
        super(ResBlock, self).__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=strides, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size
                               , stride=strides, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        x = self.bn3(self.conv3(x))
        return F.relu(out + x)


class ResSEConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p, ratio=16):
        super(ResSEConvBlock, self).__init__()
        self.conv_block = ResBlock(in_channel, out_channel, k, s, p)
        self.se = SE(out_channel, ratio)

    def forward(self, x):
        out = self.conv_block(x)
        se = self.se(out)
        out = out * se
        return out


class Scale1TPsReg(nn.Module):
    def __init__(self):
        super(Scale1TPsReg,self).__init__()
        self.line1 = nn.Sequential(
            ResBlock(2, 16, 7, 1, 3),
            ResSEConvBlock(16, 32, 7, 2, 3),
            ResSEConvBlock(32, 32, 7, 2, 3),
            ResSEConvBlock(32, 32, 7, 2, 3),
        )
        self.line2 = nn.Sequential(
            ResBlock(2, 16, 5, 1, 2),
            ResSEConvBlock(16, 32, 5, 2, 2),
        )
        self.line3 = nn.Sequential(
            ResSEConvBlock(64, 32, 3, 2, 1),
            ResSEConvBlock(32, 32, 3, 2, 1),
            ResSEConvBlock(32, 32, 3, 2, 1),
        )
        self.fc = nn.Conv2d(32, 6, 8, 1, 0)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        input = F.interpolate(input, scale_factor=0.25, mode='bilinear')
        line1_out = self.line1(input)
        line2_out = self.line2(input)
        line1_out_upsample4 = F.interpolate(line1_out, scale_factor=4, mode='bilinear')
        line3_input = torch.cat([line1_out_upsample4, line2_out], dim=1)
        line3_out = self.line3(line3_input)
        tps = self.fc(line3_out).view(-1, 2, 3)
        return tps



class Scale2TPsReg(nn.Module):
    def __init__(self):
        super(Scale2TPsReg,self).__init__()
        self.line1 = nn.Sequential(
            ResBlock(2, 32, 7, 1, 3),
            ResSEConvBlock(32, 64, 7, 2, 3),
            ResSEConvBlock(64, 64, 7, 2, 3),
            ResSEConvBlock(64, 64, 7, 2, 3),
            ResSEConvBlock(64, 64, 7, 2, 3)
        )
        self.line2 = nn.Sequential(
            ResBlock(2, 32, 5, 1, 2),
            ResSEConvBlock(32, 64, 5, 2, 2),
            ResSEConvBlock(64, 64, 5, 2, 2)
        )
        self.line3 = nn.Sequential(
            ResSEConvBlock(128, 64, 3, 2, 1),
            ResSEConvBlock(64, 64, 3, 2, 1),
            ResSEConvBlock(64, 64, 3, 2, 1),
        )
        self.fc = nn.Conv2d(64, 6, 8, 1, 0)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        input = F.interpolate(input, scale_factor=0.5, mode='bilinear')
        line1_out = self.line1(input)
        line2_out = self.line2(input)
        line1_out_upsample4 = F.interpolate(line1_out, scale_factor=4, mode='bilinear')
        line3_input = torch.cat([line1_out_upsample4, line2_out], dim=1)
        line3_out = self.line3(line3_input)
        tps = self.fc(line3_out).view(-1, 2, 3)
        return tps


class Scale3TPsReg(nn.Module):
    def __init__(self):
        super(Scale3TPsReg,self).__init__()
        self.line1 = nn.Sequential(
            ResBlock(2, 64, 7, 1, 3),
            ResSEConvBlock(64, 128, 7, 2, 3),
            ResSEConvBlock(128, 128, 7, 2, 3),
            ResSEConvBlock(128, 128, 7, 2, 3),
            ResSEConvBlock(128, 128, 7, 2, 3),
            ResSEConvBlock(128, 128, 7, 2, 3)
        )
        self.line2 = nn.Sequential(
            ResBlock(2, 64, 5, 1, 2),
            ResSEConvBlock(64, 128, 5, 2, 2),
            ResSEConvBlock(128, 128, 5, 2, 2),
            ResSEConvBlock(128, 128, 5, 2, 2)
        )
        self.line3 = nn.Sequential(
            ResSEConvBlock(256, 128, 3, 2, 1),
            ResSEConvBlock(128, 128, 3, 2, 1),
            ResSEConvBlock(128, 128, 3, 2, 1),
        )
        self.fc = nn.Conv2d(128, 6, 8, 1, 0)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        line1_out = self.line1(input)
        line2_out = self.line2(input)
        line1_out_upsample4 = F.interpolate(line1_out, scale_factor=4, mode='bilinear')
        line3_input = torch.cat([line1_out_upsample4, line2_out], dim=1)
        line3_out = self.line3(line3_input)
        tps = self.fc(line3_out).view(-1, 2, 3)
        return tps
    
    
    class DeepFeatureGenerator(nn.Module):
    def __init__(self):
        super(DeepFeatureGenerator, self).__init__()
        self.localization = ConvBlock(2, 16, 3, 1, 1)
        self.scale_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.scale_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(128),
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        self.scale_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(64),
            nn.Upsample(scale_factor=8, mode='bilinear')
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = self.localization(out)
        out_1 = self.scale_1(out)
        out_2 = self.scale_1(out)
        out_3 = self.scale_1(out)
        out = torch.cat([out, out_1, out_2, out_3], dim=1)
        out = self.attentionmechanism(out)
        return out


class ParameterRegression(nn.Module):
    def __init__(self):
        super(ParameterRegression, self).__init__()
        self.localization = nn.Sequential(
            ConvBlock(64, 64, 3, 2, 3, dilation=3),
            ConvBlock(64, 64, 3, 2, 2, dilation=2),
            ConvBlock(64, 64, 3, 2, 2, dilation=2),
            ConvBlock(64, 64, 3, 2, 1),
            ConvBlock(64, 64, 3, 2, 1),
            ConvBlock(64, 64, 3, 2, 1),
            ConvBlock(64, 64, 3, 2, 1)
        )
        self.fc = nn.Conv2d(64, 6, kernel_size=4, stride=1, padding=0)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        out = self.localization(x)
        out = self.fc(out)
        out = out.view(-1, 2, 3)
        return out


def AffineTransform(reference, sensed, affine_matrix):
    sensed_grid = F.affine_grid(affine_matrix, sensed.size())
    sensed_tran = F.grid_sample(sensed, sensed_grid)
    a = torch.tensor([[[0, 0, 1]]], dtype=torch.float)
    affine_matrix = torch.cat([affine_matrix, a], dim=1)
    inv_affine_matrix = inverse(affine_matrix)
    inv_affine_matrix_1 = inv_affine_matrix[:, 0:2, :]
    reference_grid = F.affine_grid(inv_affine_matrix_1, reference.size())
    reference_inv_tran = F.grid_sample(reference, reference_grid)
    return sensed_tran, reference_inv_tran, inv_affine_matrix


def ComputeLoss(reference, sensed_tran, sensed, reference_inv_tran, batch_size):
    loss = 0
    for i in range(batch_size):
        loss_1 = NCC_loss(reference[i].unsqueeze(0), sensed_tran[i].unsqueeze(0))
        loss_2 = NCC_loss(sensed[i].unsqueeze(0), reference_inv_tran[i].unsqueeze(0))
        loss = loss + loss_1 + loss_2
    loss = loss/batch_size
    return loss


def ComputeConstrain(matrix):
    a = matrix[:, 0, 0]
    b = matrix[:, 0, 1]
    c = matrix[:, 1, 0]
    d = matrix[:, 1, 1]
    lamda = (a+d)/torch.sqrt(4-(b-c)**2)
    constrain = torch.abs((a/lamda)**2 + b**2 + c**2 + (d/lamda)**2 - 2)/2
    return constrain


def show_plot(iteration, loss, name):
    plt.plot(iteration, loss)
    plt.savefig('./%s' % name)
    plt.show()


def ComputeError(TPs_GT, TPs_Pre):
    TPs_Pre = TPs_Pre[:, 0:2, :]
    D = TPs_Pre-TPs_GT
    return torch.sqrt(torch.sum(torch.mul(D, D))/6)

