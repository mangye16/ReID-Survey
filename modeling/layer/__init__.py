# encoding: utf-8

from .center_loss import CenterLoss
from .triplet_loss import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet
from .non_local import Non_local
from .gem_pool import GeneralizedMeanPooling, GeneralizedMeanPoolingP