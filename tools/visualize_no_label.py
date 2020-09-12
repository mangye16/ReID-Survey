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
          metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
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
            print('shape {}'.format(feat.shape))
            return feat, camid, date

    engine = Engine(_inference)

    # visualize does not have to calculate metrics
    # for name, metric in metrics.items():
    #     metric.attach(engine, name)

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
      # print(data_loader['gallery'].dataset)
      gallery_engine.run(data_loader['gallery'])
      # print(len(gallery_feat))
      gallery_feature = torch.cat(gallery_feat)

      # -------------------- visualize step ----------------------------------
      print(gallery_feature.shape)
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
      with open("./log/{}/feature-pickle.pkl".format(cfg.DATASET.NAMES), "rb").format(cfg.DATASETS.NAMES) as fout: 
        feat_dump_obj = pickle.load(fout)
        gallery_feature = feat_dump_obj["gallery"]["feat"]
        gallery_cam = feat_dump_obj["gallery"]["cam"]
        galelry_date = feat_dump_obj["gallery"]["date"]
  gallery_feature = gallery_feature.cuda()
  #######################################################################
  # sort the images
  # TODO qf - gallery_feat[i], ql -> qc , qc -> qd, gl -> gc, gc -> gd
  def sort_img(qf, qc, qd, gf, gc, gd):
      query = qf.view(-1,1)
      # print(query.shape)
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
        #   # index  index[0:2000]
        #   # not counting image from the same iden in the same cam  
        #   # good index : query label equal to gallery label
        #   query_index = np.argwhere(gl==ql)
        #   #same camera
        #   camera_index = np.argwhere(gc==qc)
        #   # here
        #   #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        #   junk_index1 = np.argwhere(gl==-1)
        #   junk_index2 = np.intersect1d(query_index, camera_index)
        #   junk_index = np.append(junk_index2, junk_index1) 
        #   # stuck
        #   mask = np.in1d(index, junk_index, invert=True)
        #   index = index[mask=]
      return index
  def imshow(path, ax,title=None):
      """Imshow for Tensor."""
      im = plt.imread(path)
      ax.imshow(im)
      if title is not None:
          ax.set_title(title)
  # TODO FIX QUERY
  def make_query(i) :
      query_ind = i
      index = sort_img(gallery_feature[i],gallery_cam[i],gallery_date[i],gallery_feature,gallery_cam,gallery_date)
      ########################################################################
      # Visualize the rank result
      _, _, _, query_path = data_loader['gallery'].dataset[i]
    #   query_lb = query_label[i]
      print('Top 10 images are as follow:')
      try: # Visualize Ranking Result 
          # Graphical User Interface is needed
          fig = plt.figure(figsize=(16,4))
          ax = fig.add_subplot(1,11,1)
          ax.axis('off')
          imshow(query_path,ax,'query')
          # show top 10
          for i in range(10):
            ax = fig.add_subplot(1,11,i+2)
            ax.axis('off')
            _, _, _, img_path = data_loader['gallery'].dataset[index[i]]
            # label = gallery_label[index[i]]
            imshow(img_path, ax)
            #   if label == query_lb:
            #       ax.set_title('%d'%(i+1), color='green')
            #   else:
            #       ax.set_title('%d'%(i+1), color='red')

            ax.set_title('%d'%(i+1))
      except RuntimeError:
          for i in range(10):
              img_path = data_loader['gallery'].dataset[index[i]][-1]
              print(img_path[0])
          print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
      plt.show()
      fig.savefig("./log/{}/query_image/show_{}.png".format(cfg.DATASETS.NAMES,query_ind))
      return fig
  i = cfg.VISUALIZE.INDEX
  if i<0 :
    # print("kaboom")
    query_size = len(data_loader["gallery"].dataset)
    for i in range(query_size) : 
      print(i)
      make_query(i)
  else :
    fig = make_query(i)
    fig.savefig("./log/{}/show.png".format(cfg.DATASETS.NAMES))
