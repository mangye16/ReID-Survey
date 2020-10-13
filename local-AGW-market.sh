# Dataset: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# last stride 1
# with center loss
# weight regularized triplet loss
# generalized mean pooling
# non local blocks
python tools/main.py --config_file='configs/AGW_baseline.yml' MODEL.DEVICE_ID "('0')" \
DATASETS.NAMES "('market1501')" OUTPUT_DIR "('./log/market1501/local-AGW-baseline')"