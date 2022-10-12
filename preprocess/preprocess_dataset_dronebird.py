from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import json

dir_name = os.path.dirname(os.path.abspath(__file__))


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '.mat').replace('data', 'annotation')
    points = loadmat(mat_path)['locations'][:, :2].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * \
        (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    print(im.size, '=>', im_h, im_w, rr)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def main(input_dataset_path, output_dataset_path, min_size=512, max_size=2048):
    for phase in ['train', 'test']:
        sub_dir = os.path.join(input_dataset_path, phase)
        if phase == 'train':
            sub_phase_list = ['train', 'val']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(output_dataset_path, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                # with open(os.path.join(dir_name, '{}.json'.format(sub_phase))) as f:
                with open(os.path.join('..', input_dataset_path, '{}.json'.format(sub_phase))) as f:
                    im_name_list = json.load(f)
                for im_name in im_name_list:
                    im_path = os.path.join('..', im_name)
                    # for i in f:
                    #     im_path = os.path.join(sub_dir, i.strip())
                    name = os.path.basename(im_path)
                    print(name)
                    im, points = generate_data(im_path, min_size, max_size)
                    im_save_path = os.path.join(sub_save_dir, name)
                    im.save(im_save_path)
                    gd_save_path = im_save_path.replace('jpg', 'npy')
                    np.save(gd_save_path, points)
        else:
            sub_save_dir = os.path.join(output_dataset_path, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            with open(os.path.join('..', input_dataset_path, 'test.json')) as f:
                im_name_list = json.load(f)
            # im_list = glob(os.path.join(sub_dir, '*jpg'))
            for im_path in im_name_list:
                name = os.path.basename(im_path)
                print(name)
                im, points = generate_data(im_path, min_size, max_size)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)


if __name__ == '__main__':
    input_dataset_path = '../../ds/dronebird'
    output_dataset_path = '../data'
    main(input_dataset_path, output_dataset_path)
