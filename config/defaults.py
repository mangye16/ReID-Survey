from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.BACKBONE = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.TRANSFER_MODE ="off"
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with weighted regularized triplet loss, options: 'on', 'off'
_C.MODEL.WEIGHT_REGULARIZED_TRIPLET = 'off'
# If train with cos 
_C.MODEL.USE_COS = False
# If train with dropout
_C.MODEL.USE_DROPOUT = False

# for baseline 
_C.MODEL.BASELINE = CN()
_C.MODEL.BASELINE.POOL_TYPE = 'avg'
_C.MODEL.BASELINE.COSINE_LOSS_TYPE = ''
_C.MODEL.BASELINE.SCALING_FACTOR = 60.0
_C.MODEL.BASELINE.MARGIN = 0.35
_C.MODEL.BASELINE.USE_BNBIAS = False 
_C.MODEL.BASELINE.USE_SESTN = False 

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image
_C.INPUT.IMG_SIZE = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
_C.INPUT.RE_MAX_RATIO = 0.4
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./toDataset')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# If use PK sampler for data loading
_C.DATALOADER.PK_SAMPLER = 'on'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3

###### Center loss  ######

_C.SOLVER.CENTER_LOSS = CN()

_C.SOLVER.CENTER_LOSS.USE = False
# # Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LOSS.LR = 0.5
_C.SOLVER.CENTER_LOSS.WEIGHT = 0.0005
# _C.SOLVER.CENTER_LOSS.ALPHA = 1.0

_C.SOLVER.CENTER_LOSS.NUM_FEATS = 2048

##########################

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (30, 55)

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'on','off'
_C.TEST.RE_RANKING = 'off'
# Path to trained model
_C.TEST.WEIGHT = ""
# Whether feature is nomalized before test, if on, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'on'
_C.TEST.EVALUATE_ONLY = 'off'

# ---------------------------------------------------------------------------- #
# Visualize
# ---------------------------------------------------------------------------- #
_C.VISUALIZE = CN()
# option
_C.VISUALIZE.OPTION = "off"
_C.VISUALIZE.CAM_OPTION = "allow_other"
_C.VISUALIZE.IMS_PER_BATCH = 256
_C.VISUALIZE.NEED_NEW_FEAT_EMBED = "off"
_C.VISUALIZE.INDEX = 0
_C.VISUALIZE.TOP_RANK = 10
_C.VISUALIZE.RE_RANK = "off"
# ---------------------------------------------------------------------------- #
# Embedding projector
# ---------------------------------------------------------------------------- #
_C.EMBEDDING_PROJECTOR = CN()
# option 
_C.EMBEDDING_PROJECTOR.OPTION = "off"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
