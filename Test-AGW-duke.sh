# Dataset: dukemtmc
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# last stride 1
# with center loss
# weight regularized triplet loss
# generalized mean pooling
# non local blocks
# without re-ranking: add TEST.RE_RANKING "('on')" for re-ranking
python3 tools/main.py --config_file='configs/AGW_baseline.yml' MODEL.DEVICE_ID "('1')" \
DATASETS.NAMES "('dukemtmc')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('./pretrained/dukemtmc_AGW.pth')" TEST.EVALUATE_ONLY "('on')" OUTPUT_DIR "('./log/Test')"
