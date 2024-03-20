import novelty_detector
from defaults import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/fmnist.yaml')


data_list = [
    {
        "inliner_classes": [7],
        "ic": 7
    }

]

total_classes = 100
mul = 0.2
folds = 5

for folding_id in range(5):  # Loop from 0 to 4
    for data in data_list:
        inliner_classes = data["inliner_classes"]
        ic = data["ic"]

        novelty_detector.main(
            folding_id,
            inliner_classes, ic,
            total_classes,
            mul,
            folds, cfg
        )
