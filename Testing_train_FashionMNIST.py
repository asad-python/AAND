import logging
from train_AAE import train
from defaults import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/fmnist.yaml')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# inliner_classes can be a list of class indices that you are considering as inliers.
inliner_classes = [7]
ic = 7  # this can be an identifier for the model or any specific configuration.

# Number of folds
folds = 5  # For MNIST, 5 folds are generated.

for folding_id in range(folds):  # Looping over each fold and training the model.
    logger.info(f"Starting training for fold {folding_id}...")
    print(ic)
    train(folding_id, inliner_classes, ic, cfg)
    logger.info(f"Training for fold {folding_id} completed.")
