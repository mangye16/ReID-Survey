python ./tools/main.py --config_file='configs/AGW_baseline.yml' \
    VISUALIZE.INDEX "(10)" MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')"  \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('./log/market1501/local-AGW-baseline/resnet50_nl_model_120.pth')" \
    VISUALIZE.OPTION "('on')" \
    OUTPUT_DIR "('./log/Test')" 
    