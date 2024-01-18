import novelty_detector
from defaults import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/coil-100.yaml')

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
