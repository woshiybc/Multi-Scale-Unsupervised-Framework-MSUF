from __future__ import print_function
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import numpy as np
from PIL import Image
from numpy import sin, cos, tan
import os
import random
from torch import linalg


def save_tensor_to_image(T, path):
    T = T.squeeze(0)
    # T_numpy = torch.tensor(T, dtype=torch.uint8).permute([1, 2, 0]).detach().cpu().numpy()
    T_numpy = torch.tensor(T, dtype=torch.uint8).squeeze(0).detach().cpu().numpy()
    T_PIL = Image.fromarray(T_numpy)
    T_PIL.save(path)


def affine(img, move_x, move_y, scale_x, scale_y, rotate_angle, shearx_angle, sheary_angle):
    img = img.unsqueeze(0)
    [b, c, w, h] = img.shape
    sx = 1/scale_x
    sy = 1/scale_y
    dx = move_x
    dy = move_y
    theta = rotate_angle*np.pi/180
    faix = shearx_angle*np.pi/180
    faiy = sheary_angle*np.pi/180
    A = torch.tensor(np.float32(np.array([[
        (sx * (cos(theta) - sin(theta) * tan(faiy)), sx * (cos(theta) * tan(faix) - sin(theta)), dx),
        (sy * (sin(theta) + cos(theta) * tan(faiy)), sy * (sin(theta) * tan(faix) + cos(theta)), dy)]]))).to(device)
    B = torch.cat([A, torch.Tensor([[[0, 0, 1]]]).to(device)], dim=1)
    Inv = linalg.inv(B)
    Inv = Inv[:, 0:2, :]
    # A = torch.tensor(np.float32(np.array([[(1, 1, 0),
    #                                        (0, 1.5, 0)]])))
    grid = F.affine_grid(A, img.size())
    img_flow = F.grid_sample(img, grid)
    matrix = A.reshape(6)
    matrix_inv = Inv.reshape(6)
    return img_flow, matrix, matrix_inv


def get_filename(path, filetype):
    name = []
    for root,dirs,files in os.walk(path):
        for i in files:
            if os.path.splitext(i)[1]==filetype:
                name.append(i)
    return name


def pil_to_tensor(p):
    # rgb = torch.tensor(np.float32(np.array(p))).to(device)
    # rgb = rgb.permute(2, 0, 1)
    p = p.convert('L')
    t = torch.tensor(np.array(p)).to(device).unsqueeze(0)
    t = t.float()
    return t


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_file_path = ''
    sen_file_path = ''
    ref_file_name = get_filename(ref_file_path, '.jpg')
    sen_file_name = get_filename(sen_file_path, '.jpg')
    assert len(ref_file_name) == len(sen_file_name)
    save_path = ''
    imgpair_label_save_path = os.path.join(save_path, 'imgpair.txt')
    with open(imgpair_label_save_path, "a") as file:
        for i in range(len(ref_file_name)):
            img_name = ref_file_name[i]
            ref_pil = Image.open(os.path.join(ref_file_path, img_name))
            ref_tensor = pil_to_tensor(ref_pil)
            sen_pil = Image.open(os.path.join(sen_file_path, img_name))
            sen_tensor = pil_to_tensor(sen_pil)
            count = 0
            img_name_no_suffix = img_name.split('.')[0]
            for j in range(4):
                ref_rotate_expand = rotate(ref_tensor, 90*j)
                sen_rotate_expand = rotate(sen_tensor, 90*j)
                for k in range(4):
                    rotate_angle = random.randrange(-10, 10, 1)
                    shearx_angle = random.randrange(-3, 3, 1)
                    sheary_angle = random.randrange(-3, 3, 1)
                    scale_x = random.randrange(8, 12, 1) / 10
                    scale_y = random.randrange(8, 12, 1) / 10
                    move_x = 0
                    move_y = 0
                    sen_wrapped, matrix, matrix_inv = affine(sen_rotate_expand, move_x, move_y, scale_x, scale_y, rotate_angle,
                                                           shearx_angle, sheary_angle)
                    count = count + 1
                    save_img_name = '%s_%d.jpg' % (img_name_no_suffix, count)
                    tps = np.array(matrix_inv.cpu())
                    save_gt_tps = '%f %f %f %f %f %f' % (tps[0], tps[1], tps[2], tps[3], tps[4], tps[5])
                    save_content = '%s %s %s\n' % (save_img_name, save_img_name, save_gt_tps)
                    file.write(save_content)
                    save_tensor_to_image(ref_rotate_expand, os.path.join(save_path, 'reference', save_img_name))
                    save_tensor_to_image(sen_wrapped, os.path.join(save_path, 'sensed', save_img_name))
