import novelty_detector
from defaults import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/mnist.yaml')

folding_id = 0
inliner_classes = [0, 1]
ic = 0
total_classes = 10
mul = 0.2
folds = 5

novelty_detector.main(
    folding_id,
    inliner_classes,ic,
    total_classes,
    mul,
    folds, cfg
)
