python ./tools/main.py --config_file='configs/AGW_baseline.yml' \
   MODEL.DEVICE_ID "('0')"   \
   DATASETS.NAMES "('oxygen')"   \
   MODEL.TRANSFER_MODE "('on')"   \
   MODEL.PRETRAIN_CHOICE "('self')"  \
   MODEL.PRETRAIN_PATH "('./log/market1501/Experiment-AGW-baseline/resnet50_nl_model_160.pth')"