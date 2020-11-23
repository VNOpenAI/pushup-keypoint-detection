import time
import numpy as np
import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import densenet169, resnet50
import torchvision.transforms.functional as functional
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest50, resnest101, resnest200, resnest269, resnest50_fast_4s2x40d
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast
from albumentations.pytorch import ToTensor
from albumentations import KeypointParams
from PIL import Image

class finetune_data(Dataset):
  def __init__(self, labels_json, folder='./filter_data/images', transform=None, sub_trans=None, mode = 'train'):
    #"operation" can be 'resize' or 'crop'
    self.transform = transform
    self.sub_trans = sub_trans
    self.mode = mode
    with open(labels_json, 'r') as fr:
      raw_labels = json.load(fr)['labels']
    self.labels = [aa for aa in raw_labels if len(aa['points']) == 7 and aa['is_pushing_up'] and aa['contains_person']]
    self.folder = folder
    self.n_data = len(self.labels)
    print('total length data: {}'.format(self.n_data))
  def __len__(self):
    return self.n_data
  def __getitem__(self, idx):
    label = self.labels[idx]
    img_path = os.path.join(self.folder, label['image'])
    # print(img_path)
    img = cv2.imread(img_path)
    # print(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    o_h, o_w = img.shape[:2]
    masks = np.zeros((15, 15, 3*(len(label['points']))), dtype = np.float32)
    raw_label = np.array(label['points'])
    # raw_label = np.concatenate([raw_label[:3], raw_label[4:]], axis=0)
    if o_h > o_w:
      new_img = np.zeros((o_h, o_h, 3), np.uint8)
      cl = ((o_h-o_w)//2)
      new_img[:,cl:cl+o_w] = img
      raw_label[:,0] = raw_label[:,0] + cl
    else:
      new_img = np.zeros((o_w, o_w, 3), np.uint8)
      cl = ((o_w-o_h)//2)
      new_img[cl:cl+o_h,:] = img
      raw_label[:,1] = raw_label[:,1] + cl
    if self.transform:
      if self.mode == 'train':
        loop = True
        count = 0
        while loop:
          aug = self.transform(image = img, keypoints = raw_label)
          nimg = aug['image']
          ncoors = aug['keypoints']
          nncoors = np.array(ncoors)/225
          # print(np.max(nncoors), np.min(nncoors))
          count += 1
          if np.all(nncoors<1) and np.all(nncoors>=0):
            img=nimg
            # print(img.dtype, img.max(), img.min())
            raw_label=np.array(ncoors).astype(np.int32)
            loop=False
            break
          elif count > 50:
            aug = self.sub_trans(image = img)
            img = aug['image']
            raw_label = (np.array(raw_label)/max([o_w, o_h]))*225
            break
      else:
        aug = self.transform(image = img)
        img = aug['image']
        raw_label = (np.array(raw_label)/max([o_w, o_h]))*225
    for i,key_point in enumerate(raw_label):
        coor_x = np.clip(int(key_point[0]), 0, 224)
        coor_y = np.clip(int(key_point[1]), 0, 224)
        os_vect_x = coor_x - (coor_x//15)*15
        os_vect_y = coor_y - (coor_y//15)*15
        masks[coor_y//15, coor_x//15, i] = 1.0
        masks[coor_y//15, coor_x//15, i+7] = os_vect_x/15
        masks[coor_y//15, coor_x//15, i+14] = os_vect_y/15
    # if self.transform:
    #   img = self.transform(img)
    return img, transforms.ToTensor()(masks)

def get_score(model, ckp_path, loader, img_size=(225,225), type='regression'):
  checkpoint=torch.load(ckp_path)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.load_state_dict(checkpoint)
  model.eval()
  correct_count=0.0
  count_all = 0
  ova_loss = 0.0
  time_takes = []
  with torch.no_grad() as tng:
    for i, data in enumerate(loader):
      count_all += data[0].shape[0]*14
      imgs, targets = data[0].to(device), data[1].to(device)
      st = time.time()
      preds = model(imgs)
      en = time.time()
      time_takes.append(en-st)
      preds = preds.cpu().numpy()
      targets = targets.cpu().numpy()
      if type == 'regression':
          err = np.abs(preds-targets)
          ova_loss += np.sum(err)
      else:
        for z, pred in enumerate(preds):
            for j,pred_sub in enumerate(pred[:7]):
            cx = np.argmax(pred_sub)%pred_sub.shape[0]
            cy = np.argmax(pred_sub)//pred_sub.shape[0]
            ovx = pred[j+7][cy,cx]*15
            ovy = pred[j+14][cy,cx]*15
            x_p = (cx*15+ovx)/img_size[1]
            y_p = (cy*15+ovy)/img_size[0]

            cx = np.argmax(targets[z,j])%targets[z,j].shape[0]
            cy = np.argmax(targets[z,j])//targets[z,j].shape[0]
            ovx = targets[z][j+7][cy,cx]*15
            ovy = targets[z][j+14][cy,cx]*15
            x_t = (cx*15+ovx)/img_size[1]
            y_t = (cy*15+ovy)/img_size[0]

            ova_loss += np.abs(x_t-x_p) + np.abs(y_t-y_p)
  
  return ova_loss/count_all, 1/np.mean(time_takes)

test_json = './test.json'
img_size = (225,225)
batch_size = 32

sub_trans = Compose([Resize(img_size[0], img_size[1]), ToTensor()])
trans = Compose([
                 Resize(img_size[0], img_size[1]),
                 RandomBrightnessContrast(always_apply=True, brightness_limit=0.25, contrast_limit=0.25),
                 ShiftScaleRotate(0.0, 0.05, 5, always_apply=True),
                 ToTensor(),], keypoint_params=KeypointParams(format='xy', remove_invisible=False))

data_test = finetune_data(test_json, transform=sub_trans, mode='test')
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=4)