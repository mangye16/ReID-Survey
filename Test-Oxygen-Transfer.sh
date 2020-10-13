python tools/main.py --config_file='configs/AGW_baseline.yml' \
 MODEL.DEVICE_ID "('0')" \
 DATASETS.NAMES "('oxygen')" \
 MODEL.PRETRAIN_CHOICE "('self')" \
 TEST.WEIGHT "('./log/oxygen1/Experiment-AGW-baseline/resnet50_nl_model_120.pth')" \
 TEST.EVALUATE_ONLY "('on')" \
 OUTPUT_DIR "('./log/Oxygen_Test')"