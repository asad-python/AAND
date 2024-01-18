
import logging
import os
from train_AAE import train
from defaults import get_cfg_defaults
import random

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/coil-100.yaml')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# List of inliner_classes and their corresponding ic values
# Example given below, you can extend or modify this list as per your requirements
data_list = [
    {
        "inliner_classes": [58, 29, 15, 7, 75, 37, 62],
        "ic": 5829157753762
    }
    # You can add more configurations like this:
    # {"inliner_classes": [23, 78, 65], "ic": 237865}
]

for folding_id in range(5):  # Loop from 0 to 4
    for data in data_list:
        inliner_classes = data["inliner_classes"]
        ic = data["ic"]

        # inlier class set index (used to save model with unique filename)
        print(ic)
        train(folding_id, inliner_classes, ic, cfg)
