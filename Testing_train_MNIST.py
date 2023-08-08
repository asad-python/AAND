import logging
import os
from train_AAE import train
from defaults import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/mnist.yaml')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

folding_id = 0  # Id of the fold. For MNIST, 5 folds are generated, so folding_id must be in range [0..5]
inliner_classes = [0, 1]  # List of classes considered inliers
ic = 0  # inlier class set index (used to save model with unique filename)

train(folding_id, inliner_classes, ic, cfg)


