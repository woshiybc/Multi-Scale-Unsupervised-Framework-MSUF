from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import inverse
from losses import NCC_loss
from generation import save_tensor_to_image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import network


def read_img_and_gt(imgpairtxt_path):
    img_and_gt = []
    with open(imgpairtxt_path, "r") as f:
        for line in f.readlines():
            data = line.split('\n\t')
            for str in data:
                sub_str = str.split(' ')
            if sub_str:
                img_and_gt.append(sub_str)
    return img_and_gt


def pil_to_tensor(p):
    if p.mode != 'L':
        p = p.convert('L')
    return torch.tensor(np.float32(np.array(p))).to(device).unsqueeze(0)


# def ComputeConstrain(matrix):
#     a = matrix[:, 0, 0]
#     b = matrix[:, 0, 1]
#     c = matrix[:, 1, 0]
#     d = matrix[:, 1, 1]
#     lamda = (a+d)/torch.sqrt(4-(b-c)**2)
#     constrain = torch.abs((a/lamda)**2 + b**2 + c**2 + (d/lamda)**2 - 2)/2
#     return constrain


def ComputeLoss(reference, sensed_tran, sensed, reference_inv_tran):
    loss_1 = NCC_loss(reference, sensed_tran)
    loss_2 = NCC_loss(sensed, reference_inv_tran)
    loss = loss_1 + loss_2
    return loss


def AffineTransform(reference, sensed, affine_matrix):
    sensed_grid = F.affine_grid(affine_matrix, sensed.size())
    sensed_tran = F.grid_sample(sensed, sensed_grid)
    a = torch.tensor([[[0, 0, 1]]], dtype=torch.float).to(device)
    a = a.repeat(sensed.size()[0], 1, 1)
    affine_matrix = torch.cat([affine_matrix, a], dim=1)
    inv_affine_matrix = inverse(affine_matrix)
    inv_affine_matrix_1 = inv_affine_matrix[:, 0:2, :]
    reference_grid = F.affine_grid(inv_affine_matrix_1, reference.size())
    reference_inv_tran = F.grid_sample(reference, reference_grid)
    return sensed_tran, reference_inv_tran, inv_affine_matrix


def show_plot(iteration, loss, name):
    plt.plot(iteration, loss)
    plt.savefig('./%s' % name)
    plt.show()


class MyDataSet(Dataset):
    def __init__(self, mode, data_path):
        self.mode = mode
        if self.mode == 'train':
            self.data_path = os.path.join(data_path, 'train')
            self.imginfo_path = os.path.join(self.data_path, 'imgpair.txt')
            self.img_and_gt = read_img_and_gt(self.imginfo_path)
            self.ref_path = os.path.join(self.data_path, 'reference')
            self.sen_path = os.path.join(self.data_path, 'sensed')

        if self.mode == 'test':
            self.data_path = os.path.join(data_path, 'test')
            self.imginfo_path = os.path.join(self.data_path, 'imgpair.txt')
            self.img_and_gt = read_img_and_gt(self.imginfo_path)
            self.ref_path = os.path.join(self.data_path, 'reference')
            self.sen_path = os.path.join(self.data_path, 'sensed')

    def __len__(self):
        return len(self.img_and_gt)

    def __getitem__(self, index):
        self.ref_name = self.img_and_gt[index][0]
        self.sen_name = self.img_and_gt[index][1]
        self.gt_tps = list(map(float, self.img_and_gt[index][2:8]))
        self.ref_pil = Image.open(os.path.join(self.ref_path, self.ref_name))
        self.sen_pil = Image.open(os.path.join(self.sen_path, self.sen_name))
        self.ref_tensor = pil_to_tensor(self.ref_pil)
        self.sen_tensor = pil_to_tensor(self.sen_pil)
        self.gt_tps_tensor = torch.Tensor(self.gt_tps).to(device)
        return self.ref_tensor, self.sen_tensor, self.gt_tps_tensor, self.sen_name


def test():
    time_test_start = time.time()
    print('Using device ' + str(device) + ' for testing!')
    scale_1_model.eval()
    scale_2_model.eval()
    scale_3_model.eval()
    Loss = []
    for i, data in enumerate(dataloader):
        ref_tensor = data[0]
        sen_tensor = data[1]
        # gt_tps = data[2]
        sen_name = data[3]
        scale_1_affine_parameter = scale_1_model(ref_tensor, sen_tensor)
        sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_1 = AffineTransform(ref_tensor, sen_tensor,
                                                                                  scale_1_affine_parameter)
        loss_1 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
        scale_2_affine_parameter = scale_2_model(ref_tensor, sen_tran_tensor)
        sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_2 = AffineTransform(ref_tensor, sen_tran_tensor,
                                                                                  scale_2_affine_parameter)
        loss_2 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
        scale_3_affine_parameter = scale_3_model(ref_tensor, sen_tran_tensor)
        sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_3 = AffineTransform(ref_tensor, sen_tran_tensor,
                                                                                  scale_3_affine_parameter)
        for j in range(len(sen_tran_tensor)):
            save_sensed_corrected_name = sen_name[j]
            save_tensor_to_image(sen_tran_tensor[j], os.path.join(save_sensed_corrected_path, save_sensed_corrected_name))
        # inv_affine_parameter = torch.matmul(torch.matmul(inv_affine_parameter_1, inv_affine_parameter_2),
        #                                     inv_affine_parameter_3)
        loss_3 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
        loss = 0.14285714 * loss_1 + 0.28571429 * loss_2 + 0.57142857 * loss_3
        pp = loss.detach().cpu()
        if not np.isnan(pp):
            loss.backward()
            Loss.append(loss.detach().cpu())
        if i % 5 == 0:
            print('%f%% loss: %f' % (i/total_epoch*100, loss))
    average_loss = np.mean(Loss)
    print('average loss is %f' % average_loss)
    print("Total test time: {:.1f}s".format(time.time() - time_test_start))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """"""
    mode = 'test'
    data_path = ''
    save_sensed_corrected_path = os.path.join(data_path, mode, 'save_sensed_corrected')
    batch_size = 8
    constrain_weight = 0.2
    scale_1_model = torch.load('')
    scale_2_model = torch.load('')
    scale_3_model = torch.load('')
    """"""
    dataset = MyDataSet(mode, data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    total_epoch = len(dataloader)
    if mode == 'test':
        test()