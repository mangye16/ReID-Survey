# GO TO ROOT DIRECTORY
mkdir ./toDataset/oxygen1
mkdir ./toDataset/oxygen1/gallery
#  Then you put oxygen folder in gallery folder
#  And the structure must be like /gallery/mbk-12-4/*.jpg
mkdir ./log/oxygen1
# and run this command in single line
!python3 ./tools/main.py --config_file='configs/AGW_baseline.yml' \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('oxygen1')" \  
    MODEL.PRETRAIN_CHOICE "('self')" \ 
    TEST.WEIGHT "('YOUR_SAVED_TRAINED_MODEL_PATH')" \ 
    VISUALIZE.OPTION "('on_no_label')" \
    VISUALIZE.INDEX "(12)" 
    /
