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
python3 tools/main.py --config_file='configs/AGW_baseline.yml' MODEL.DEVICE_ID "('2')" \
DATASETS.NAMES "('dukemtmc')" OUTPUT_DIR "('./log/dukemtmc/Experiment-AGW-baseline')"