# Dataset: train on market1501, eval on partial_reid and partial_ilids
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# last stride 1
# with center loss
# weight regularized triplet loss
# generalized mean pooling
# non local blocks
python3 tools/main.py --config_file='configs/AGW_baseline.yml' MODEL.DEVICE_ID "('1')" \
DATASETS.NAMES "('market1501')" TEST.PARTIAL_REID "('on')" OUTPUT_DIR "('./log/market1501/Experiment-AGW-baseline-partial')"
