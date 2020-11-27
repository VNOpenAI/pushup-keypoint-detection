import torch, time, os, shutil, cv2
import torchvision
from torchvision.models import densenet169, resnet50
import torch.nn.functional as F
from utils_t import build_detection_based_model, build_regression_based_model, preprocessed_img_test, heatmap2coor
import numpy as np

class SHPE_model():
    def __init__(self, pb_type='detection', model_name='resnest', n_kps=7, metrics=None, define_model=None, define_img_size=None):
        self.pb_type = pb_type
        self.model_name = model_name
        self.n_kps = n_kps
        if pb_type == 'detection':
            self.model = build_detection_based_model(model_name, n_kps)
            self.img_size = (225,225)
        elif pb_type == 'regression':
            self.model = build_regression_based_model(model_name, n_kps)
            self.img_size = (224,224)
        elif pb_type == 'define':
            if define_model is None:
                raise Exception("not define model!!!")
            self.model = define_model
            self.img_size = define_img_size
        else:
            raise Exception("not support this pb_type!!!")
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, loader_dict, loss_func, optimizer, lr=3e-4, use_lr_sch=False, epochs=120, ckp_dir='./checkpoint', writer=None):
        # criterion = loss_func
        metrics = self.metrics
        metrics['loss'] = loss_func
        optimizer = optimizer(self.model.parameters(),lr)
        if 'train' not in list(loader_dict.keys()):
            raise Exception("missing \'train\' keys in loader_dict!!!")
        if use_lr_sch:
            lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 80, 1/3)
        else:
            lr_sch = None
        best_loss = 80.0
        if os.path.exists(ckp_dir):
            shutil.rmtree(ckp_dir)
        os.mkdir(ckp_dir)
        modes = list(loader_dict.keys())
        for epoch in range(epochs):
            s="Epoch [{}/{}]:".format(epoch+1, epochs)
            start = time.time()
            for mode in modes:
                running_metrics = dict.fromkeys(metrics.keys(), 0.0)
                ova_len = loader_dict[mode].dataset.n_data
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for i, data in enumerate(loader_dict[mode]):
                    imgs, labels = data[0].to(self.device), data[1].to(self.device)
                    preds = self.model(imgs)
                    loss = metrics['loss'](preds, labels)
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    iter_len = imgs.size()[0]
                    # preds = (preds > 0.5).float()
                    running_loss = loss.item()*iter_len/ova_len
                    for key in list(metrics.keys()):
                        if key == 'loss':
                            running_metrics[key] += running_loss
                        else:
                            running_metrics[key] += metrics[key](preds, labels).item()*iter_len/ova_len
                if writer is not None:
                    for key in list(metrics.keys()):
                        writer.add_scalars(key, {mode: running_metrics['key']}, epoch)
                for key in list(metrics.keys()):
                    s += "{}_{} {:.3f} -".format(mode, key, running_metrics[key])
            end = time.time()
            s = s[:-1] + "({:.1f}s)".format(end-start)
            print(s)
            # if writer is not None:
            #     n_visual = 4
            #     preds = self.predict(imgs[:n_visual])
            #     imgs_new = imgs[:n_visual].cpu().numpy()
            #     imgs_new = np.stack([imgs_new[:,0], imgs_new[:,1], imgs_new[:,2]], axis=-1)
            #     preds = (preds*self.img_size[:2]).astype(np.int32)
            #     for z, img_new in enumerate(imgs_new):
            #         cv2.polylines(img_new, [preds[z]], True, (0,0,255), 2)
            #     imgs_new = np.stack([imgs_new[...,0], imgs_new[...,1], imgs_new[...,2]], axis=1)
            #     imgs_new = torch.from_numpy(imgs_new)
            #     img_grid = torchvision.utils.make_grid(imgs_new)
            #     writer.add_image(str(epoch+1)+'_visual_img', img_grid)
            if running_metrics['loss'] < best_loss or (epoch+1)%10==0:
                best_loss = running_metrics['loss']
                torch.save(self.model.state_dict(), os.path.join(ckp_dir,'epoch'+str(epoch+1)+'.pt'))
                print('new checkpoint saved!')
            if lr_sch is not None:
                lr_sch.step()
                print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))

    def load_ckp(self, ckp_path):
        checkpoint=torch.load(ckp_path)
        self.model.load_state_dict(checkpoint)

    def predict(self, img):
        self.model.eval()
        with torch.no_grad() as tng:
            preds = self.model(img)
            preds = preds.cpu().numpy()
            if self.pb_type == 'regression':
                preds = np.stack([preds[:,::2], preds[:,1:][:,::2]], axis=-1)
            elif self.pb_type == 'detection':
                preds = heatmap2coor(preds, self.n_kps, self.img_size)
        return preds

    def predict_raw(self, img_in):
        img, dmax, clx, cly = preprocessed_img_test(img_in, self.img_size)
        img = img.to(self.device)
        preds = self.predict(img)
        preds = (preds[0]*dmax).astype(np.int32) - np.array([clx, cly])
        return preds
    
    def pred_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 50, (frame_width, frame_height))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                preds = self.predict_raw(frame)
                cv2.polylines(frame, [preds], True, (0,0,255), 2)
                out.write(frame)
            else:
                break
        cap.release()
        out.release()

    def evaluate(self, loader):
        self.model.eval()
        with torch.no_grad() as tng:
            ova_len = loader.dataset.n_data
            ova_loss = 0
            for i, data in enumerate(loader):
                imgs, targets = data[0].to(self.device), data[1].to(self.device)
                preds = self.model(imgs)
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                if self.pb_type == 'regression':
                    ova_loss += np.sum(np.abs(preds-targets))
                elif self.pb_type == 'detection':
                    preds = heatmap2coor(preds, self.n_kps, self.img_size)
                    targets = heatmap2coor(targets, self.n_kps, self.img_size)
                    ova_loss += np.sum(np.abs(preds-targets))
                else:
                    return None
        return ova_loss/(ova_len*14)