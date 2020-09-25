# GO TO ROOT DIRECTORY
# rm -rf ./toDataset/oxygen1
# rm -rf ./toDataset/oxygen1/gallery
# mkdir ./toDataset/oxygen1
# mkdir ./toDataset/oxygen1/gallery
#  Then you put oxygen folder in gallery folder
#  And the structure must be like /gallery/mbk-12-4/*.jpg
# rm -rf ./log/oxygen1
# rm -rf ./log/oxygen1/query_image
# mkdir ./log/oxygen1
# mkdir ./log/oxygen1/query_image

# and run this command in single line
python ./tools/main.py --config_file='configs/AGW_baseline.yml' \
  MODEL.DEVICE_ID "('0')" \
  DATASETS.NAMES "('oxygen1')" \
  MODEL.PRETRAIN_CHOICE "('self')" \
  TEST.WEIGHT "('./log/market1501/Experiment-AGW-baseline/resnet50_nl_model_160.pth')" \
  VISUALIZE.OPTION "('on_no_label')" \
  VISUALIZE.INDEX "(-1)" \
  VISUALIZE.NEED_NEW_FEAT_EMBED "('off')" \
  VISUALIZE.TOP_RANK "(20)" \
  VISUALIZE.RE_RANK "('on')" 
