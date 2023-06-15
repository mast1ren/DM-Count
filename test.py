import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19
import scipy.io as sio

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument(
    '--crop-size', type=int, default=512, help='the crop size of the train image'
)
parser.add_argument(
    '--model-path',
    type=str,
    default='../../nas-public-linkdata/ds/result/dmc/ckpts/input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/best_model_2.pth',
    help='saved model path',
)
parser.add_argument(
    '--data-path',
    type=str,
    default='./preprocessed_data',
    help='saved model path',
)
parser.add_argument(
    '--dataset',
    type=str,
    default='dronebird',
    help='dataset name: qnrf, nwpu, sha, shb',
)
parser.add_argument(
    '--pred-density-map-path',
    type=str,
    default='',
    help='save predicted density maps when pred-density-map-path is not empty.',
)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda')

model_path = args.model_path
crop_size = args.crop_size
data_path = args.data_path
if args.dataset.lower() == 'qnrf':
    dataset = crowd.Crowd_qnrf(
        os.path.join(data_path, 'test'), crop_size, 8, method='val'
    )
elif args.dataset.lower() == 'nwpu':
    dataset = crowd.Crowd_nwpu(
        os.path.join(data_path, 'val'), crop_size, 8, method='val'
    )
elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
    dataset = crowd.Crowd_sh(
        os.path.join(data_path, 'test_data'), crop_size, 8, method='val'
    )
elif args.dataset.lower() == 'dronebird':
    dataset = crowd.Crowd_dronebird(
        os.path.join(data_path, 'test'), crop_size, 8, method='val'
    )
else:
    raise NotImplementedError
dataloader = torch.utils.data.DataLoader(
    dataset, 1, shuffle=False, num_workers=1, pin_memory=False
)

if args.pred_density_map_path:
    import cv2

    if not os.path.exists(args.pred_density_map_path):
        os.makedirs(args.pred_density_map_path)


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


model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
image_errs = []
i = 0
preds = []
gts = []
preds_hist = [[] for i in range(10)]
gts_hist = [[] for i in range(10)]
attri = [
    "sunny",
    "backlight",
    "crowded",
    "sparse",
    "60",
    "90",
    "stand",
    "fly",
    "small",
    "mid",
]
for inputs, count, name in dataloader:
    inputs = inputs.to(device)
    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    with torch.set_grad_enabled(False):
        outputs, _ = model(inputs)
    img_err = count[0].item() - torch.sum(outputs).item()

    print(
        '\r[{:>{}}/{}] img: {}, error: {}, gt: {}, pred: {}'.format(
            i,
            len(str(len(dataloader))),
            len(dataloader),
            name,
            img_err,
            count[0].item(),
            torch.sum(outputs).item(),
        ),
        end='',
    )
    image_errs.append(img_err)
    i += 1
    if args.pred_density_map_path:
        vis_img = outputs[0, 0].cpu().numpy()
        # normalize density map values from 0 to 1, then map it to 0-255.
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
        vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        cv2.imwrite(
            os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img
        )
print()

image_errs = np.abs(np.array(image_errs))
mse = np.sqrt(np.mean(np.square(image_errs)))
mae = np.mean(np.abs(image_errs))
print(
    '{}: mae {}, mse {}, min {}, max {}\n'.format(
        model_path, mae, mse, np.min(image_errs), np.max(image_errs)
    )
)
