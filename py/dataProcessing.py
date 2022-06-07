# -*- coding:utf-8 -*-
import numpy as np

def newtonNormDec(numlist, factor):
  numArray = np.array(numlist)
  newtonCool = np.exp(-factor*numArray) # 牛顿冷却法(减)

  _range = np.max(newtonCool) - np.min(newtonCool)
  norm = (newtonCool - np.min(newtonCool)) / _range

  return norm


def newtonNormInc(numlist, factor):
  numArray = np.array(numlist)
  newtonCool = np.exp(factor*numArray) # 牛顿冷却法（增）

  _range = np.max(newtonCool) - np.min(newtonCool)
  norm = (newtonCool - np.min(newtonCool)) / _range

  return norm
