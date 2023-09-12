import logging
import os
from train_AAE import train
from defaults import get_cfg_defaults
import random


cfg = get_cfg_defaults()
cfg.merge_from_file('configs/coil-100.yaml')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

folding_id = 3  # Id of the fold. For MNIST, 5 folds are generated, so folding_id must be in range [0..5]
#inliner_classes = [23,78,65]
#inliner_classes = [53, 70, 55, 9]
#ic = 237865
#inliner_classes = random.sample(range(1, 101), 7)
inliner_classes = [33]
ic = 33
# inlier class set index (used to save model with unique filename)
print(ic)
train(folding_id, inliner_classes, ic, cfg)


