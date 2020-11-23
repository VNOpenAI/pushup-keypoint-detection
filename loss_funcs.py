from torch import nn
import torch

class Regression_based_Loss(nn.Module):
    def init(self):
        super(Regression_based_Loss, self).init()

    def forward(self, pred, target, mse_w = 10, angle_w = None, regularize_w = None, epsilon=1e-5):
        coor_x_t = target[:][:,::2]
        coor_y_t = target[:,1:][:,::2]
        coor_x_p = pred[:][:,::2]
        coor_y_p = pred[:,1:][:,::2]
        mse = torch.mean(torch.mean((coor_x_t-coor_x_p)**2 + (coor_y_t-coor_y_p)**2, dim=1))
        ova_loss = mse_w*mse
        if angle_w not None:
            ova_loss += angle_w*angle_loss(coor_x_p, coor_y_p, coor_x_t, coor_y_t, epsilon)
        if regularize_w not None:
            ova_loss += angle_w*angle_loss(coor_x_p, coor_y_p, coor_x_t, coor_y_t, epsilon)
        return 5*angle_loss + 5*reg_loss

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
    def init(self):
        super(Detection_base_Loss, self).init()
    def forward(self, pred, target, n_kps=7, hm_w=4, os_w=1):
        hm_pred = pred[:,:n_kps]
        hm_target = target[:,:n_kps]
        lmap = (hm_target == 1.0)*1.0
        lmap = torch.cat([lmap, lmap], dim=1)
        heatmap_loss = F.binary_cross_entropy(hm_pred, hm_target, reduction='mean')
        coor_pred = pred[:,n_kps:]*lmap
        coor_target = target[:,n_kps:]*lmap
        offset_loss = 1/(2*n_kps*pred.shape[0])*F.mse_loss(coor_pred, coor_target, reduction='sum')
        return hm_w*heatmap_loss + os_w*offset_loss