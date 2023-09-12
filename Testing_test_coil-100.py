import novelty_detector
from defaults import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/coil-100.yaml')

folding_id = 3  # Id of the fold. For MNIST, 5 folds are generated, so folding_id must be in range [0..5]
inliner_classes = [33]
ic = 33
total_classes = 100
mul = 0.2
folds = 5

novelty_detector.main(
    folding_id,
    inliner_classes,ic,
    total_classes,
    mul,
    folds, cfg
)
