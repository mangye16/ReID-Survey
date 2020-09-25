# encoding: utf-8
import logging
import torchvision
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from utils.reid_metric import r1_mAP_mINP, r1_mAP_mINP_reranking
from ignite.handlers import Timer
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from utils.re_ranking import re_ranking_no_label

global ITER
ITER = 0
gallery_feat = []
gallery_cam = []
gallery_date = []
from torch.utils.tensorboard import SummaryWriter
        
def create_feature_extractor(model, device=None):
    """
      Factory function for creating an evaluator for supervised models

      Args:
          model (`torch.nn.Module`): the model to evaluate
          device (str, optional): device type specification (default: None).
              Applies to both model and batches.
      Returns:
          Engine: an evaluator engine with supervised inference function
    """

    def _inference(engine, batch):
        global ITER
        model.eval()
        with torch.no_grad():
            data, camid, date = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            
            feat = model(data)
            # print('shape {}'.format(feat.shape))
            return feat, camid, date

    engine = Engine(_inference)

    return engine


def do_visualize_no_label(
        cfg,
        model,
        data_loader
):
  if ( not os.path.exists('./log/{}/feature-pickle.pkl'.format(cfg.DATASETS.NAMES)) or cfg.VISUALIZE.NEED_NEW_FEAT_EMBED == "on" )  :
      print("compute new feature embedding")
      global gallery_feat, gallery_cam, gallery_date   
      device = cfg.MODEL.DEVICE
      logger = logging.getLogger("reid_baseline")
      logger.info("Enter inferencing to visualize no label data")

      print("Create gallery engine to make feature extractor")
      gallery_engine = create_feature_extractor(model,
                                              device=device)

      
      @gallery_engine.on(Events.ITERATION_COMPLETED)
      def append_result_gal(engine) :
        global gallery_feat, gallery_cam, gallery_date
        global ITER
        ITER += 1
        gallery_feat.append(gallery_engine.state.output[0])
        gallery_cam.extend(gallery_engine.state.output[1])
        gallery_date.extend(gallery_engine.state.output[2])
        logger.info("Epoch[{}] Iteration[{}/{}] output shape : {}"
                          .format(engine.state.epoch, ITER, len(data_loader['gallery']), gallery_engine.state.output[0].shape))

      #Show result
      gallery_engine.run(data_loader['gallery'])
      # print(type(gallery_feat))
      gallery_feature = torch.cat(gallery_feat)

      # -------------------- visualize step ----------------------------------
      # print(gallery_feature.shape)
      if(not os.path.isdir("./log/{}".format(cfg.DATASETS.NAMES))) :
        creating_directory = "./log/{}".format(cfg.DATASETS.NAMES)
        os.mkdir(creating_directory)
      with open("./log/{}/feature-pickle.pkl".format(cfg.DATASETS.NAMES), "wb") as fout:
          feat_dump_obj = {
            "gallery" : {
                "feat" : gallery_feature, 
                "cam" : gallery_cam,
                "date" : gallery_date
            }
          }
          pickle.dump(feat_dump_obj, fout, protocol=pickle.HIGHEST_PROTOCOL)
  else :
      with open("./log/{}/feature-pickle.pkl".format(cfg.DATASETS.NAMES), "rb") as fout: 
        feat_dump_obj = pickle.load(fout)
        gallery_feature = feat_dump_obj["gallery"]["feat"]
        gallery_cam = feat_dump_obj["gallery"]["cam"]
        galelry_date = feat_dump_obj["gallery"]["date"]
  gallery_feature = gallery_feature.cuda()
  #######################################################################
  # sort the images by cosine similarity matrix score
  # TODO qf - gallery_feat[i], ql -> qc , qc -> qd, gl -> gc, gc -> gd
  def sort_img(qf, qc, qd, gf, gc, gd,ignore_index=None):
      query = qf.view(-1,1)
      gf = torch.cat([gf[0:ignore_index], gf[ignore_index+1:]])
      score = torch.mm(gf,query)
      # tensor.cuda() is used to move a tensor to GPU memory.
      # tensor.cpu() moves it back to memory accessible to the CPU.
      score = score.squeeze(1).cpu()
      score = score.numpy()
      # predict index
      index = np.argsort(score)  #from small to large
      index = index[::-1]
      #   TODO NOMASK FOR NOW
      #   TODO MASK FOR ITSELF
      #   # not counting image from the same iden in the same cam
      # good index : query label equal to gallery label
      # query_index = np.argwhere(gl==ql)
      # diff date
      date_index = np.argwhere(gd==qd)
      # same camera
      camera_index = np.argwhere(gc==qc)

      #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
      # junk_index1 = np.argwhere(gl==-1)
      junk_index2 = np.intersect1d(date_index, camera_index)
      junk_index = np.append(junk_index2, junk_index1) 
      # stuck
      mask = np.in1d(index, junk_index, invert=True)
      index = index[mask]
      return index
  def imshow(path, ax,title=None):
      """Imshow for Tensor."""
      im = plt.imread(path)
      ax.imshow(im)
      if title is not None:
          ax.set_title(title)
  def extract_iden(path) :
    return path.split("\\")[-1].split("_")[-1].split("-")[0]
  # TODO FIX QUERY
  def make_query(i,re_rank=False,reranking_list=None,cam_option=None) :
      query_ind = i
      re_rank_str = ""
      if not re_rank :
        index = sort_img(gallery_feature[i],gallery_cam[i],gallery_date[i],gallery_feature,gallery_cam,gallery_date,ignore_index=i)
      else :
        re_rank_str = "_re_rank_"
        index = list(reranking_list[query_ind])
      index= list(index)
      # print(query_ind)
      # print(index[:20])
      index.remove(query_ind)
      # print(index[:20])
      # exit()
      ########################################################################
      # Visualize the rank result
      iden_set = set()
      _, _, _, query_path = data_loader['gallery'].dataset[query_ind]
      query_iden = extract_iden(query_path)
      iden_set.add(query_iden)
      # print(query_path.split("\\")[-1].split("_")[-1].spilt("-")[0])
      #   query_lb = query_label[i]
      # print('Top 10 images are as follow:')
      try: # Visualize Ranking Result 
          # Graphical User Interface is needed
          fig = plt.figure(figsize=(16,4))
          ax = fig.add_subplot(1,11,1)
          ax.axis('off')
          imshow(query_path,ax,'query '+query_iden)
          # show top 10
          ind = 0
          index_ind = 0
          while ind < 10:
            _, _, _, img_path = data_loader['gallery'].dataset[index[index_ind]]
            index_ind += 1
            img_iden = extract_iden(img_path)
            if img_iden not in iden_set :
              iden_set.add(img_iden)
            else :
              continue
            ax = fig.add_subplot(1,11,ind+2)
            ax.axis('off')
            # print(index[ind])
            # print(query_path==img_path)
            imshow(img_path, ax,str(ind+1)+"_"+img_iden)
            ind += 1
            # ax.set_title('%d'%(ind+1))
      except RuntimeError:
          for i in range(10):
              img_path = data_loader['gallery'].dataset[index[i]][-1]
          print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
      # plt.show()
      if(not os.path.isdir("./log/{}/query_image".format(cfg.DATASETS.NAMES))) :
        creating_directory = "./log/{}/query_image".format(cfg.DATASETS.NAMES)
        os.mkdir(creating_directory)
      if cam_option == "none" :
        cam_option=""
      fig.savefig("./log/{}/query_image/show_{}{}{}.png".format(cfg.DATASETS.NAMES,query_ind,re_rank_str,cam_option))
      return fig,re_rank_str
  i = cfg.VISUALIZE.INDEX
  # query all image in gallery
  is_re_rank = False
  if cfg.VISUALIZE.RE_RANK == "on" :
    # print(gallery_feature)
    # print(gallery_feature.shape)
    # gallery_feature = gallery_feature.view(-1,1)
    reranking_list = re_ranking_no_label(gallery_feature,k1=20,k2=6,lambda_value=0.3)
    is_re_rank = True
  cam_option = cfg.VISUALIZE.CAM_OPTION
  print("start querying")
  if i<0 :
    query_size = len(data_loader["gallery"].dataset)
    for i in range(query_size) : 
      make_query(i, is_re_rank,reranking_list,cam_option=cam_option)
  else :
    fig, re_rank_str = make_query(i, is_re_rank, reranking_list)
    fig.savefig("./log/{}/show{}.png".format(cfg.DATASETS.NAMES,re_rank_str))