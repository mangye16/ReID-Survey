from __future__ import absolute_import

from .models import *

__factory = {
    'AGW_Plus_Baseline': AGW_Plus_Baseline,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
