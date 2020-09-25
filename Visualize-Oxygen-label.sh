python ./tools/main.py --config_file='configs/AGW_baseline.yml' \
    VISUALIZE.INDEX "(-1)" \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('oxygen')"  \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('./log/oxygen/Experiment-AGW-baseline/resnet50_nl_model_120.pth')" \
    VISUALIZE.OPTION "('on')" \
    OUTPUT_DIR "('./log/Oxygen/visualize_label_log.txt')" 
    