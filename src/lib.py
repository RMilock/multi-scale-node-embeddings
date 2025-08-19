import pandas as pd
import numpy as np
import os
import sys
import torch as tc

from scipy.optimize import minimize
import scipy.sparse as sp
from scipy.special import expit # for LPCA