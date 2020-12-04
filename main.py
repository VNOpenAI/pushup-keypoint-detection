import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from models import SHPE_model
from torchsummary import summary
from simple_baselines.pose_resnet import get_pose_net
from stacked_hourglass.posenet import PoseNet
from deep_high_resolution.pose_hrnet import get_pose_net as gpn
from deep_high_resolution.pose_resnet import get_pose_net as gpnn

shpe_model = SHPE_model('detection', 'resnest', stride=15)
# shpe_model.load_ckp('checkpoints/efficientb2_heatmap_filter_2211.pt')
# summary(shpe_model.model)
# shpe_model.pred_live()
# model = get_pose_net(False)
# model = PoseNet()
# model = gpn()
# model = gpnn()
summary(model)
print(model)