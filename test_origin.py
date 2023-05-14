import argparse
from PIL import Image
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19
import json
from torchvision import transforms
import h5py
from sklearn.metrics import mean_squared_error, mean_absolute_error
from glob import glob
import scipy.io as sio


def get_seq_class(seq, set):
    backlight = [
        'DJI_0021',
        'DJI_0022',
        'DJI_0032',
        'DJI_0202',
        'DJI_0339',
        'DJI_0340',
        'DJI_0463',
        'DJI_0003',
    ]

    fly = [
        'DJI_0177',
        'DJI_0174',
        'DJI_0022',
        'DJI_0180',
        'DJI_0181',
        'DJI_0200',
        'DJI_0544',
        'DJI_0012',
        'DJI_0178',
        'DJI_0343',
        'DJI_0185',
        'DJI_0195',
        'DJI_0996',
        'DJI_0977',
        'DJI_0945',
        'DJI_0946',
        'DJI_0091',
        'DJI_0442',
        'DJI_0466',
        'DJI_0459',
        'DJI_0464',
    ]

    angle_90 = [
        'DJI_0179',
        'DJI_0186',
        'DJI_0189',
        'DJI_0191',
        'DJI_0196',
        'DJI_0190',
        'DJI_0070',
        'DJI_0091',
    ]

    mid_size = [
        'DJI_0012',
        'DJI_0013',
        'DJI_0014',
        'DJI_0021',
        'DJI_0022',
        'DJI_0026',
        'DJI_0028',
        'DJI_0028',
        'DJI_0030',
        'DJI_0028',
        'DJI_0030',
        'DJI_0034',
        'DJI_0200',
        'DJI_0544',
        'DJI_0463',
        'DJI_0001',
        'DJI_0149',
    ]

    light = 'sunny'
    bird = 'stand'
    angle = '60'
    size = 'small'
    # resolution = '4k'
    if seq in backlight:
        light = 'backlight'
    # elif seq in cloudy:
    #     light = 'cloudy'
    if seq in fly:
        bird = 'fly'
    if seq in angle_90:
        angle = '90'
    if seq in mid_size:
        size = 'mid'

    # if seq in uhd:
    #     resolution = 'uhd'

    count = 'sparse'
    loca = sio.loadmat(
        os.path.join(
            '../../nas-public-linkdata/ds/dronebird/',
            set,
            'ground_truth',
            'GT_img' + str(seq[-3:]) + '000.mat',
        )
    )['locations']
    if loca.shape[0] > 150:
        count = 'crowded'
    # return light, resolution, count
    return light, angle, bird, size, count


parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
# parser.add_argument('--crop-size', type=int, default=512,
#                     help='the crop size of the train image')
# parser.add_argument('--model-path', type=str, default='pretrained_models/model_qnrf.pth',
#                     help='saved model path')
# parser.add_argument('--data-path', type=str,
#                     default='data/QNRF-Train-Val-Test',
#                     help='saved model path')
# parser.add_argument('--dataset', type=str, default='dronebird',
#                     help='dataset name: qnrf, nwpu, sha, shb')
# parser.add_argument('--pred-density-map-path', type=str, default='',
#                     help='save predicted density maps when pred-density-map-path is not empty.')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda')

model_path = '../../nas-public-linkdata/ds/result/dmc/ckpts/input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/best_model_2.pth'
crop_size = 512

test_path = './preprocessed_data/test'
# test_files = os.listdir(test_path)
# test_files = [os.path.join(test_path, file) for file in test_files if file.endswith('.jpg')]
# with open('../../ds/dronebird/test.json') as f:
#     dataset = json.load(f)

# dataset = sorted(glob(os.path.join(test_path, '*.jpg')))
with open("../../nas-public-linkdata/ds/dronebird/test.json") as f:
    dataset = json.load(f)

model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
image_errs = []

preds = [[] for i in range(10)]
gts = [[] for i in range(10)]
i = 0
# temp = './yapd/train/img012000.jpg'
# tempimg = Image.open(temp).convert('RGB')
# tempinput = transforms.ToTensor()(tempimg)
# tempinput = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tempinput).unsqueeze(0)
# tempinput = tempinput.to(device)
# with torch.set_grad_enabled(False):
#     outputs, _ = model(tempinput)
#     pre_density = outputs[0, 0].cpu().numpy()
#     np.save('./pre_density.npy', pre_density)

# for inputs, count, name in dataloader:
for img_path in dataset:
    img_path = os.path.join('../../nas-public-linkdata/ds/dronebird', img_path)
    # seq = img_path.split('/')[-3]
    seq = int(os.path.basename(img_path)[3:6])
    seq = 'DJI_' + str(seq).zfill(4)
    light, angle, bird, size, count = get_seq_class(seq, 'test')
    img = Image.open(img_path).convert('RGB')
    inputs = transforms.ToTensor()(img)
    inputs = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
        inputs
    ).unsqueeze(0)
    inputs = inputs.to(device)

    gt_path = (
        img_path.replace('images', 'ground_truth')
        .replace('jpg', 'mat')
        .replace('img', 'GT_img')
    )
    gt = sio.loadmat(gt_path)['locations'].shape[0]
    # gt_path = img_path.replace('jpg', 'npy')
    # gt = np.load(gt_path).shape[0]
    # count = 'crowded' if gt > 150 else 'sparse'

    # gt_path = img_path.replace('data', 'annotation').replace('jpg', 'h5')
    # gt = np.array(h5py.File(gt_path, 'r')['density']).sum()

    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    with torch.set_grad_enabled(False):
        outputs, _ = model(inputs)
    pred_e = torch.sum(outputs).item()
    gt_e = gt
    img_err = abs(gt - torch.sum(outputs).item())
    if light == 'sunny':
        preds[0].append(pred_e)
        gts[0].append(gt_e)
    elif light == 'backlight':
        preds[1].append(pred_e)
        gts[1].append(gt_e)
    # else:
    #     preds[2].append(pred_e)
    #     gts[2].append(gt_e)
    if count == 'crowded':
        preds[2].append(pred_e)
        gts[2].append(gt_e)
    else:
        preds[3].append(pred_e)
        gts[3].append(gt_e)
    if angle == '60':
        preds[4].append(pred_e)
        gts[4].append(gt_e)
    else:
        preds[5].append(pred_e)
        gts[5].append(gt_e)
    if bird == 'stand':
        preds[6].append(pred_e)
        gts[6].append(gt_e)
    else:
        preds[7].append(pred_e)
        gts[7].append(gt_e)
    if size == 'small':
        preds[8].append(pred_e)
        gts[8].append(gt_e)
    else:
        preds[9].append(pred_e)
        gts[9].append(gt_e)
    print(
        '\r[{:>{}}/{}] img: {}, error: {}, gt: {}, pred: {}'.format(
            i,
            len(str(len(dataset))),
            len(dataset),
            os.path.basename(img_path),
            img_err,
            gt,
            torch.sum(outputs).item(),
        ),
        end='',
    )
    image_errs.append(img_err)
    i += 1
    # if args.pred_density_map_path:
    #     vis_img = outputs[0, 0].cpu().numpy()
    #     # normalize density map values from 0 to 1, then map it to 0-255.
    #     vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    #     vis_img = (vis_img * 255).astype(np.uint8)
    #     vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    #     cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img)
print()

image_errs = np.abs(np.array(image_errs))
mse = np.sqrt(np.mean(np.square(image_errs)))
mae = np.mean(np.abs(image_errs))
with open('test.txt', 'w') as f:
    f.write(
        '{}: mae {}, mse {}, min {}, max {}\n'.format(
            model_path, mae, mse, np.min(image_errs), np.max(image_errs)
        )
    )
    print(
        '{}: mae {}, mse {}, min {}, max {}\n'.format(
            model_path, mae, mse, np.min(image_errs), np.max(image_errs)
        )
    )

    attri = [
        'sunny',
        'backlight',
        'crowded',
        'sparse',
        '60',
        '90',
        'stand',
        'fly',
        'small',
        'mid',
    ]
    for i in range(10):
        # print(len(preds[i]))
        if len(preds[i]) == 0:
            continue
        print(
            '{}: MAE:{}. RMSE:{}.'.format(
                attri[i],
                mean_absolute_error(preds[i], gts[i]),
                np.sqrt(mean_squared_error(preds[i], gts[i])),
            )
        )
        f.write(
            '{}: MAE:{}. RMSE:{}.\n'.format(
                attri[i],
                mean_absolute_error(preds[i], gts[i]),
                np.sqrt(mean_squared_error(preds[i], gts[i])),
            )
        )

    # if i == 3:
    #     error = np.abs(np.array(preds[i]) - np.array(gts[i]))
    #     loss_rate = error / np.array(gts[i])
    #     print('sparse: loss_rate:{}. max:{}, min:{}.'.format(np.mean(loss_rate), np.max(loss_rate), np.min(loss_rate)))
