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

model_path = './ckpts/input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/best_model_7.pth'
crop_size = 512

with open('../../ds/dronebird/test.json') as f:
    dataset = json.load(f)

model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
image_errs = []
i = 0
# for inputs, count, name in dataloader:
for img_path in dataset:
    img = Image.open(img_path).convert('RGB')
    inputs = transforms.ToTensor()(img)
    inputs = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inputs).unsqueeze(0)
    inputs = inputs.to(device)

    gt_path = img_path.replace('data', 'annotation').replace('jpg', 'h5')
    gt = np.array(h5py.File(gt_path, 'r')['density']).sum()
    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    with torch.set_grad_enabled(False):
        outputs, _ = model(inputs)
    img_err = abs(gt-torch.sum(outputs).item())

    print('\r[{:>{}}/{}] img: {}, error: {}, gt: {}, pred: {}'.format(i, len(str(len(dataset))), len(dataset), os.path.basename(img_path), img_err, gt, torch.sum(outputs).item()), end='')
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
print('{}: mae {}, mse {}, min {}, max {}\n'.format(model_path, mae, mse, np.min(image_errs), np.max(image_errs)))
