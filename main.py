import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from models import SHPE_model
from torchsummary import summary

shpe_model = SHPE_model('detection', 'efficient')
shpe_model.load_ckp('checkpoints/efficientb2_heatmap_filter_2211.pt')
summary(shpe_model.model)
shpe_model.pred_live()