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
import tqdm

global ITER
ITER = 0
query_feat = []
query_cam = []
query_label = []
gallery_feat = []
gallery_cam = []
gallery_label = []
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
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            
            feat = model(data)
            # print('shape {}'.format(feat.shape))
            return feat, pids, camids

    engine = Engine(_inference)

    # visualize does not have to calculate metrics
    # for name, metric in metrics.items():
    #     metric.attach(engine, name)

    return engine


def do_visualize(
        cfg,
        model,
        data_loader,
        num_query
):
  if ( not os.path.exists('./log/{}/feature-pickle.pkl'.format(cfg.DATASETS.NAMES))  or cfg.VISUALIZE.NEED_NEW_FEAT_EMBED == "on" )  :
      print("compute new feature embedding")
      global query_feat, query_cam, query_label
      global gallery_feat, gallery_cam, gallery_label   
      #   writer = SummaryWriter('./log/{}/Experiment-AGW-baseline/test_image').format(cfg.DATASET.NAMES)
      device = cfg.MODEL.DEVICE
      logger = logging.getLogger("reid_baseline")
      logger.info("Enter inferencing to visualize")

      print("Create query engine and gallery engine to make feature extractor")
      query_engine = create_feature_extractor(model,
                                              device=device)
      gallery_engine = create_feature_extractor(model,
                                              device=device)

      # timer = Timer(average=True)
      # timer.attach(query_engine, pause=Events.ITERATION_COMPLETED)
      
      @gallery_engine.on(Events.ITERATION_COMPLETED)
      def append_result_gal(engine) :
        global gallery_feat, gallery_cam, gallery_label
        global ITER
        ITER += 1
        gallery_feat.append(gallery_engine.state.output[0])
        gallery_cam.extend(gallery_engine.state.output[2])
        gallery_label.extend(gallery_engine.state.output[1])
        # logger.info("Epoch[{}] Iteration[{}/{}] output shape : {}"
        #                   .format(engine.state.epoch, ITER, len(data_loader['query']), query_engine.state.output[0].shape))

      @query_engine.on(Events.ITERATION_COMPLETED)
      def append_result_query(engine) :
        global query_feat, query_cam, query_label
        global ITER
        ITER += 1
        query_feat.append(query_engine.state.output[0])
        query_cam.extend(query_engine.state.output[2])
        query_label.extend(query_engine.state.output[1])
      #Show result

      query_engine.run(data_loader['query'])
      # print(torch.cat(query_feat).shape)
      gallery_engine.run(data_loader['gallery'])
      # print(torch.cat(gallery_feat).shape)

      query_feature = torch.cat(query_feat)
      gallery_feature = torch.cat(gallery_feat)

      # -------------------- visualize step ----------------------------------
      print(query_feature.shape)
      print(gallery_feature.shape)
      with open("./log/{}/feature-pickle.pkl".format(cfg.DATASETS.NAMES), "wb") as fout:
          feat_dump_obj = {
            "query" : {
                "feat" : query_feature, 
                "id" : query_label,
                "cam" : query_cam
              },
            "gallery" : {
                "feat" : gallery_feature, 
                "id" : gallery_label,
                "cam" : gallery_cam
            }
          }
          pickle.dump(feat_dump_obj, fout, protocol=pickle.HIGHEST_PROTOCOL)
  else :
      with open("./log/{}/feature-pickle.pkl".format(cfg.DATASETS.NAMES), "rb") as fout: 
        feat_dump_obj = pickle.load(fout)
        query_feature = feat_dump_obj["query"]["feat"]
        query_label = feat_dump_obj["query"]["id"]
        query_cam = feat_dump_obj["query"]["cam"]
        gallery_feature = feat_dump_obj["gallery"]["feat"]
        gallery_label = feat_dump_obj["gallery"]["id"]
        gallery_cam = feat_dump_obj["gallery"]["cam"]
  query_feature = query_feature.cuda()
  gallery_feature = gallery_feature.cuda()
  #######################################################################
  # sort the images
  def sort_img(qf, ql, qc, gf, gl, gc):
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
      # index = index[0:2000]
      # not counting image from the same iden in the same cam  
      # good index : query label equal to gallery label
      query_index = np.argwhere(gl==ql)
      #same camera
      camera_index = np.argwhere(gc==qc)
      # here
      #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
      junk_index1 = np.argwhere(gl==-1)
      junk_index2 = np.intersect1d(query_index, camera_index)
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
  def make_query(i) :
      query_ind = i
      index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
      ########################################################################
      # Visualize the rank result
      _, _, _, query_path = data_loader['query'].dataset[i]
      query_lb = query_label[i]
    #   print('Top 10 images are as follow:')
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
              gallery_img, _, _, img_path = data_loader['gallery'].dataset[index[i]]
              label = gallery_label[index[i]]
              imshow(img_path, ax)
              if label == query_lb:
                  ax.set_title('%d'%(i+1), color='green')
              else:
                  ax.set_title('%d'%(i+1), color='red')
      except RuntimeError:
          for i in range(10):
              img_path = data_loader['gallery'].dataset[index[i]][-1]
              print(img_path[0])
          print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    #   plt.show()
      fig.savefig("./log/{}/query_image/show_{}.png".format(cfg.DATASETS.NAMES,query_ind))
      return fig
  i = cfg.VISUALIZE.INDEX
  if i<0 :
    # print("kaboom")
    query_size = len(data_loader["query"].dataset)
    for i in tqdm.tqdm(range(query_size)) : 
    #   print(i)
      make_query(i)
  else :
    fig = make_query(i)
    fig.savefig("./log/{}/show.png".format(cfg.DATASETS.NAMES))
