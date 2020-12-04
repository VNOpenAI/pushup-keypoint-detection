import numpy as np
def heatmap2coor_numpy(hp_preds, n_kps = 7, img_size=(225,225)):
    heatmaps = hp_preds[:,:n_kps]
    flatten_hm = heatmaps.reshape((heatmaps.shape[0], n_kps, -1))
    flat_vectx = hp_preds[:,n_kps:2*n_kps].reshape((heatmaps.shape[0], n_kps, -1))
    flat_vecty = hp_preds[:,2*n_kps:].reshape((heatmaps.shape[0], n_kps, -1))
    flat_max = np.argmax(flatten_hm, axis=-1)
    max_mask = flatten_hm == np.expand_dims(np.max(flatten_hm, axis=-1), axis=-1)
    cxs = flat_max%(heatmaps.shape[-2])
    cys = flat_max//(heatmaps.shape[-2])
    ovxs = np.sum(flat_vectx*max_mask, axis=-1)
    ovys = np.sum(flat_vectx*max_mask, axis=-1)
    xs_p = (cxs*15+ovxs)/img_size[1]
    ys_p = (cys*15+ovys)/img_size[0]
    hp_preds = np.stack([xs_p, ys_p], axis=-1)
    return hp_preds

def pcks_score(pred, target, pb_type='regression', n_kps=7, img_size=(225,225), id_shouder=(3,5), thresh=0.4, stride=None):
    if pb_type == 'detection' and stride is None:
        raise Exception("missing \'stride\' param on detection problem")
    sr = id_shouder[0]
    sl = id_shouder[1]
    ova_len = len(pred)*n_kps
    if pb_type == 'regression':
        shouders_len = ((target[...,sr:sr+1]-target[...,sl:sl+1])**2 + (target[...,sr+n_kps:sr+n_kps+1]-target[...,sl+n_kps:sl+n_kps+1])**2)**0.5
        err = np.abs(pred-target)
        err = (err[...,:n_kps]**2 + err[...,n_kps]**2)**0.5
        err = np.sum(err < shouders_len*thresh)
    elif pb_type == 'detection':
        pred = heatmap2coor(pred, n_kps, img_size, stride)
        target = heatmap2coor(target, n_kps, img_size, stride)
        shouders_len = ((target[:,sr:sr+1,0]-target[:,sl:sl+1,0])**2 + (target[:,sr:sr+1,1]-target[:,sl:sl+1,1])**2)**0.5
        err = np.abs(pred-target)
        err = (err[...,0]**2 + err[...,1]**2)**0.5
        err = np.sum(err < shouders_len*thresh)
    else:
        return None
    return err/ova_len