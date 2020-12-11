from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import heatmap2coor

class Regression_based_Loss(nn.Module):
    def __init__(self, mse_w = 10, angle_w = None, regularize_w = None, epsilon=1e-5):
        super(Regression_based_Loss, self).__init__()
        self.mse_w = mse_w
        self.angle_w = angle_w
        self.regularize_w = regularize_w
        self.epsilon = epsilon

    def forward(self, pred, target):
        coor_x_t = target[:][:,::2]
        coor_y_t = target[:,1:][:,::2]
        coor_x_p = pred[:][:,::2]
        coor_y_p = pred[:,1:][:,::2]
        mse = torch.mean(torch.mean((coor_x_t-coor_x_p)**2 + (coor_y_t-coor_y_p)**2, dim=1))
        ova_loss = self.mse_w*mse
        if self.angle_w is not None:
            ova_loss += self.angle_w*self.angle_loss(coor_x_p, coor_y_p, coor_x_t, coor_y_t, self.epsilon)
        if self.regularize_w is not None:
            ova_loss += self.regularize_w*self.regularize_loss(coor_x_p, coor_y_p, self.epsilon)
        return ova_loss

    def angle_loss(self, coor_x_p, coor_y_p, coor_x_t, coor_y_t, epsilon=1e-5):
        ra1_t = torch.atan2((coor_y_t[:,1] - coor_y_t[:,0]), (coor_x_t[:,1] - coor_x_t[:,0] + epsilon))
        ra1_p = torch.atan2((coor_y_p[:,1] - coor_y_p[:,0]), (coor_x_p[:,1] - coor_x_p[:,0] + epsilon))
        ra2_t = torch.atan2((coor_y_t[:,2] - coor_y_t[:,1]), (coor_x_t[:,2] - coor_x_t[:,1] + epsilon))
        ra2_p = torch.atan2((coor_y_p[:,2] - coor_y_p[:,1]), (coor_x_p[:,2] - coor_x_p[:,1] + epsilon))
        la1_t = torch.atan2((coor_y_t[:,-2] - coor_y_t[:,-1]), (coor_x_t[:,-2] - coor_x_t[:,-1] + epsilon))
        la1_p = torch.atan2((coor_y_p[:,-2] - coor_y_p[:,-1]), (coor_x_p[:,-2] - coor_x_p[:,-1] + epsilon))
        la2_t = torch.atan2((coor_y_t[:,-3] - coor_y_t[:,-2]), (coor_x_t[:,-3] - coor_x_t[:,-2] + epsilon))
        la2_p = torch.atan2((coor_y_p[:,-3] - coor_y_p[:,-2]), (coor_x_p[:,-3] - coor_x_p[:,-2] + epsilon))
        angle_loss = torch.mean(((ra1_t - ra1_p)/(8*np.pi))**2+((ra2_t - ra2_p)/(8*np.pi))**2+((la1_t - la1_p)/(8*np.pi))**2+((la2_t - la2_p)/(8*np.pi))**2)
        return angle_loss

    def regularize_loss(self, coor_x_p, coor_y_p, epsilon=1e-5):
        wrist_coor = torch.atan2((coor_y_p[:,0] - coor_y_p[:,-1]), (coor_x_p[:,0] - coor_x_p[:,-1] + epsilon))
        elbow_coor = torch.atan2((coor_y_p[:,1] - coor_y_p[:,-2]), (coor_x_p[:,1] - coor_x_p[:,-2] + epsilon))
        shouder_coor = torch.atan2((coor_y_p[:,2] - coor_y_p[:,-3]), (coor_x_p[:,2] - coor_x_p[:,-3] + epsilon))
        regularize_loss = torch.mean(((wrist_coor-elbow_coor)/(6*np.pi))**2+((elbow_coor-shouder_coor)/(6*np.pi))**2+((shouder_coor-wrist_coor)/(6*np.pi))**2)
        return regularize_loss

class Detection_based_Loss(nn.Module):
    def __init__(self, n_kps=7, hm_w=4, os_w=1):
        super(Detection_based_Loss, self).__init__()
        self.n_kps =n_kps
        self.hm_w = hm_w
        self.os_w = os_w
    def forward(self, pred, target):
        hm_pred = pred[:,:self.n_kps]
        hm_target = target[:,:self.n_kps]
        lmap = (hm_target == 1.0)*1.0
        lmap = torch.cat([lmap, lmap], dim=1)
        heatmap_loss = F.binary_cross_entropy(hm_pred, hm_target, reduction='mean')
        coor_pred = pred[:,self.n_kps:]*lmap
        coor_target = target[:,self.n_kps:]*lmap
        offset_loss = 1/(2*self.n_kps*pred.shape[0])*F.mse_loss(coor_pred, coor_target, reduction='sum')
        return self.hm_w*heatmap_loss + self.os_w*offset_loss

class MultiObject_Loss(nn.Module):
    def __init__(self, n_kps=7, hm_w=4, os_w=1, cls_w=4):
        super(Detection_based_Loss, self).__init__()
        self.n_kps =n_kps
        self.hm_w = hm_w
        self.os_w = os_w
        self.cls_w = cls_w
    def forward(self, pred, target):
        cls_pred, sub_pred = pred
        cls_target, sub_target = target
        cls_weights = torch.where(target == 1.0, 1.6, 0.4)
        hm_pred = sub_pred[:,:self.n_kps]
        hm_target = sub_target[:,:self.n_kps]
        lmap = (hm_target == 1.0)*1.0
        lmap = torch.cat([lmap, lmap], dim=1)
        heatmap_loss = F.binary_cross_entropy(hm_pred, hm_target, reduction='mean')
        coor_pred = sub_pred[:,self.n_kps:]*lmap
        coor_target = sub_target[:,self.n_kps:]*lmap
        offset_loss = 1/(2*self.n_kps*pred.shape[0])*F.mse_loss(coor_pred, coor_target, reduction='sum')
        cls_loss = torch.mean(F.binary_cross_entropy(cls_pred, cls_target, reduction=None)*cls_weights)
        return self.hm_w*heatmap_loss + self.os_w*offset_loss + self.cls_w*cls_loss

class MAE(nn.Module):
    def __init__(self, pb_type='detection', n_kps=7, img_size=(225, 225), stride=None):
        super(MAE, self).__init__()
        self.n_kps =n_kps
        self.pb_type = pb_type
        self.img_size = img_size
        self.stride = stride
        if self.pb_type == 'detection' and self.stride is None:
            raise Exception("missing \'stride\' param on detection problem")
    def forward(self, pred, target):
        if self.pb_type == 'regression':
            ova_loss = torch.mean(torch.sum(torch.abs(pred-target), dim=-1)/(2*self.n_kps))
        elif self.pb_type == 'detection':
            pred = heatmap2coor(pred, self.n_kps, self.img_size, self.stride)
            target = heatmap2coor(target, self.n_kps, self.img_size, self.stride)
            ova_loss = torch.mean(torch.sum(torch.abs(pred-target), dim=(-1,-2))/(2*self.n_kps))
        else:
            return None
        return ova_loss

class PCKS(nn.Module):
    def __init__(self, pb_type='detection', n_kps=7, img_size=(225,225), id_shouder=(2,4), thresh=0.4, stride=None):
        super(PCKS, self).__init__()
        self.n_kps =n_kps
        self.pb_type = pb_type
        self.img_size = img_size
        self.sr = id_shouder[0]
        self.sl = id_shouder[1]
        self.thresh = thresh
        self.stride = stride
        if self.pb_type == 'detection' and self.stride is None:
            raise Exception("missing \'stride\' param on detection problem")
    def forward(self, pred, target):
        ova_len = len(pred)*self.n_kps
        if self.pb_type == 'regression':
            shouders_len = ((target[...,self.sr:self.sr+1]-target[...,self.sl:self.sl+1])**2 + (target[...,self.sr+self.n_kps:self.sr+self.n_kps+1]-target[...,self.sl+self.n_kps:self.sl+self.n_kps+1])**2)**0.5
            err = torch.abs(pred-target)
            err = (err[...,:self.n_kps]**2 + err[...,self.n_kps]**2)**0.5
            err = torch.sum(err < shouders_len*self.thresh)
        elif self.pb_type == 'detection':
            pred = heatmap2coor(pred, self.n_kps, self.img_size, self.stride)
            target = heatmap2coor(target, self.n_kps, self.img_size, self.stride)
            shouders_len = ((target[:,self.sr:self.sr+1,0]-target[:,self.sl:self.sl+1,0])**2 + (target[:,self.sr:self.sr+1,1]-target[:,self.sl:self.sl+1,1])**2)**0.5
            err = torch.abs(pred-target)
            err = (err[...,0]**2 + err[...,1]**2)**0.5
            err = torch.sum(err < shouders_len*self.thresh)
        else:
            return None
        return err/ova_len

class F1(nn.Module):
    def __init__(self, thresh=0.5, eps=1e-5):
        super(F1, self).__init__()
        self.thresh=0.5
        self.eps=1e-5
    def forward(self, pred, target):
        pred_thresh = (pred > 0.5)*1.0
        tp = torch.sum((pred_thresh == target)*(target==1.0))
        tn = torch.sum((pred_thresh == target)*(target==0.0))
        fp = torch.sum((pred_thresh != target)*(target==0.0))
        fn = torch.sum((pred_thresh != target)*(target==1.0))
        recall = tp / (fn+tp+self.eps)
        precision = tp / (fp+tp+self.eps)
        f1_score = 2 * precission * recall / (precision + recall + self.eps)
        return f1_score