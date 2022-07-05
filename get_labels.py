from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
import faiss

from functools import partial

import torch
import torch.nn as nn

from models import SupResNet
from utils import (
    # get_features,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_scores_one_cluster,
)
import data