python tools/main.py --config_file='configs/AGW_baseline.yml' MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('market1501')"  MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('./log/market1501/local-AGW-baseline/resnet50_nl_model_120.pth')" TEST.EVALUATE_ONLY "('on')" \
    EMBEDDING_PROJECTOR.OPTION "('on')" \
    OUTPUT_DIR "('./log/market1501/local-AGW-baseline/test_embedding_projector')"