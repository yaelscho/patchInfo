from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
import random
import shutil
import mmap
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import threading
from collections import Counter
from itertools import permutations as perm
from itertools import combinations as comb
from itertools import combinations_with_replacement, islice
from heapq import nsmallest
import scipy
import sklearn
import cv2
import pandas as pd
from collections import Counter
import cupy as cp
import image_similarity_measures
from image_similarity_measures.quality_metrics import *
import matplotlib.pyplot as plt
import sklearn.metrics
import pickle
from tqdm import tqdm
import torch
from sklearn.decomposition import PCA, IncrementalPCA
from datetime import datetime
import math
import time
from scipy.io import loadmat
import numba
from numba import jit, cuda
from numba.extending import overload
import os
from PIL import Image
import tensorflow_datasets as tfds
import json
import numpy as np
import tensorflow as tf
from six.moves import xrange
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering
from pycocotools.coco import COCO
import multiprocessing
import resnet50_input
import resnet50
import logging
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.sparse import lil_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh


import torch.nn as nn

logging.getLogger('numba').setLevel(20)

FLAGS = tf.compat.v1.app.flags.FLAGS


tf.compat.v1.app.flags.DEFINE_string('eval_dir', 'eval',"""Directory where to write event logs.""")
tf.compat.v1.app.flags.DEFINE_string('train_dir', 'chk',"""Directory where to read model checkpoints.""")
tf.compat.v1.app.flags.DEFINE_boolean('evaluation', False, "Whether this is evaluation or representation calculation")
tf.compat.v1.app.flags.DEFINE_integer('num_examples', 200, """Number of examples to run.""")

def mostCommonClass(patch):
  val, counts = np.unique(patch, return_counts=True)
  ind = np.argmax(counts)
  return val[ind] if counts[ind]/sum(counts) > 0.8 else None
#
def mostCommonClass_pascal(patch):
  val, counts = np.unique(patch.reshape(-1, 4), axis=0, return_counts=True)
  ind = np.argmax(counts)
  return val[ind] if counts[ind]/sum(counts) > 0.8 else None


# def uniGT(gt):
#   unigt = np.zeros(gt[0][0][0][0][0].shape)
#   for p in gt:
def getHist(sess, image, GT, fuse):
  pos_dist = []
  neg_dist = []
  for i in range(FLAGS.num_examples):
    im, gt, rep = sess.run([image, GT, fuse])
    rep = np.squeeze(rep)
    res_im = rep
    res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
    res_reduced = PCA(n_components=3, svd_solver='full').fit_transform(res_vec)
    rep_3 = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], 3))
    rep_3 = rep_3 - np.amin(rep_3) / np.amax(rep_3 - np.amin(rep_3))
    xVec = np.arange(0, rep.shape[0] - 1, 4)
    yVec = np.arange(0, rep.shape[1] - 1, 4)
    for x in xVec:
      for y in yVec:
        # x = np.random.randint(0, rep.shape[1]-1, 2)
        # y = np.random.randint(0, rep.shape[2]-1, 2)
        pix1 = rep[x, y, :]
        randx = np.random.choice(rep.shape[0] - 1)
        randy = np.random.choice(rep.shape[1] - 1)
        # randx = x+5
        # randy = y+5
        pix2 = rep[randx, randy, :]
        dist = np.sqrt(sum(np.power((pix1 - pix2), 2)))

        # GT = loadmat(os.sep.join(['BSDS500/data/groundTruth/test/' + filenames[i] + '.mat']))['groundTruth']
        # person1 = GT[0, 0][0, 0][0]
        # person2 = GT[0, 1][0, 0][0]
        # person3 = GT[0, 2][0, 0][0]
        # person4 = GT[0, 3][0, 0][0]
        #
        x_im = x * 4
        y_im = y * 4
        randx_im = randx * 4
        randy_im = randy * 4
        if x_im == randx_im and y_im == randy_im:
          continue

        score = bool(mostCommonClass(gt, x_im, y_im) == mostCommonClass(gt, randx_im, randy_im))
        if score:
          pos_dist.append(dist)
        else:
          neg_dist.append(dist)
  return pos_dist, neg_dist

def lda_fromScratch(X, y):
  # Compute the mean vectors for each class
  class_labels = np.unique(y)
  num_classes = len(class_labels)
  mean_vectors = []
  class_weights = []
  for label in class_labels:
      class_samples = X[y == label]
      mean_vectors.append(np.mean(class_samples, axis=0))
      class_weights.append(float(len(class_samples)) / len(X))

  # Compute the within-class scatter matrix
  Sw = np.zeros((X.shape[1], X.shape[1]))
  for label, mean in zip(class_labels, mean_vectors):
      class_samples = X[y == label]
      class_scatter = np.cov(class_samples.T)
      Sw += class_weights[label] * class_scatter
  # Sw += 1e-6 * np.identity(Sw.shape[0])

  # # addapt SW
  # alpha = 0.9
  # Sw = alpha* Sw + (1-alpha)*np.eye(Sw.shape[0])
  # Sw_inv = np.linalg.inv(Sw)
  #
  # SVD Sw
  # u, s, vh = np.linalg.svd(Sw)
  # Sw_inv = np.linalg.pinv(Sw)


  # #diagonal Sw
  for i in range(Sw.shape[0]):
    for j in range(Sw.shape[1]):
      if i != j:
        Sw[i,j] = 0
  Sw_inv = np.linalg.inv(Sw)

  mean_diff = mean_vectors[0] - mean_vectors[1]

  return Sw_inv @ mean_diff
  # # Compute the between-class scatter matrix
  # overall_mean = np.mean(X, axis=0)
  # Sb = np.zeros((X.shape[1], X.shape[1]))
  # for label, mean in zip(class_labels, mean_vectors):
  #     mean_diff = (mean - overall_mean).reshape(-1, 1)
  #     Sb += class_weights[label] * mean_diff.dot(mean_diff.T)


  # # Solve the generalized eigenvalue problem
  # eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
  # eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
  # eigen_pairs.sort(key=lambda x: x[0], reverse=True)
  #
  # # Select the eigenvectors
  # num_dims = num_classes - 1
  # selected_eigenvectors = np.column_stack([eigen_pairs[i][1] for i in range(num_dims)])

  # # Transform the data
  # X_lda = X.dot(selected_eigenvectors)

  # return selected_eigenvectors



def calcPatchDist(patch1, patch2):
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist1 = np.mean([np.amin([np.linalg.norm(pix2-pix1,2) for pix1 in patch1]) for pix2 in patch2])
  dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  return np.mean([dist1, dist2])

# @jit(parallel=True)
def distances(X, Y):
  dist = np.zeros((X.shape[0], Y.shape[0]))
  dist = (X ** 2).sum(axis=1)[:, np.newaxis] + (Y ** 2).sum(axis=1) - 2 * np.matmul(X, Y.T)
  dist[dist<0] = 0
  return dist
# @jit(parallel=True)


def distances_GPU(X, Y):
  X = cp.asarray(X)
  Y = cp.asarray(Y)

  dist = cp.sum(X ** 2, axis=1)[:, cp.newaxis] + cp.sum(Y ** 2, axis=1) - 2 * (X @ Y.T)
  dist = cp.maximum(dist, 0)  # for numerical stability instead of dist[dist<0]=0
  return dist

def calcPatchDistmulti_lib(dou):
  p1, p2 = dou
  patch1 = p1['patch']
  patch2 = p2['patch']
  return p1['idx'], p2['idx'], fsim(patch1, patch2)



def transorm2LDA(dou):
  dou, lda = dou
  p1, p2 = dou
  _, _, v, label = makeDou4LDA(dou)
  try:
    dist = lda.transform([v])
  except:
    dist =np.dot(v, lda.coef_.T)
  return p1['idx'], p2['idx'], dist.item(), label


def LDA(patches, solver='svd'):
  patches_new = random.sample(patches, len(patches))
  doues = comb(patches_new, 2)

  data = np.zeros((len(patches_new), len(patches_new), 128))
  labels = np.zeros((len(patches_new), len(patches_new)))
  with multiprocessing.Pool(processes=16) as p:
    with tqdm(total=(len(patches_new) ** 2) / 2) as pbar:
      for d in p.imap_unordered(makeDou4LDA, doues):
        pbar.update()
        i = np.where([p['idx']==d[0] for p in patches_new])
        j = np.where([p['idx']==d[1] for p in patches_new])
        data[i, j, :] = d[2]
        data[j, i, :] = d[2]
        labels[i, j] = d[3]
        labels[j, i] = d[3]
  # same = data[labels==1]
  # notsame = data[labels==0]
  # cov_same = np.cov(same, rowvar=False, bias=True)
  # cov_notsame = np.cov(notsame, rowvar=False, bias=True)
  # w_same = len(same)/(len(same)+len(notsame))
  # w_notsame = 1 - w_same
  # weightedCov = w_same*cov_same + w_notsame*cov_notsame
  try:
    lda = LinearDiscriminantAnalysis(n_components=1, store_covariance=True, solver=solver, shrinkage='auto')
    data = data.reshape((len(patches_new) * len(patches_new), 128))
    return lda.fit(data, labels.reshape(len(patches_new) * len(patches_new)))
  except:
    lda = LinearDiscriminantAnalysis(n_components=1, store_covariance=True, solver=solver)

  data = data.reshape((len(patches_new)*len(patches_new), 128))
  return lda.fit(data, labels.reshape(len(patches_new)*len(patches_new)))


def makeDou4LDA(dou):
  p1, p2 = dou
  patch1 = p1['patch']
  patch2 = p2['patch']
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist = np.sqrt(np.around(distances(patch1, patch2),3))
  dist1_vec = np.amin(dist, axis=0)
  dist2_vec = np.amin(dist, axis=1)
  return p1['idx'], p2['idx'], sorted(np.hstack((dist1_vec, dist2_vec))), int(p1['seg']==p2['seg'])


# @jit(target_backend='cuda')
def calcPatchDistmulti(dou):
  # for dou in tqdm(doues):
  p1, p2 = dou
  patch1 = p1['patch']
  patch2 = p2['patch']
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist = np.sqrt(np.around(distances(patch1, patch2),3))
  dist1_vec = np.amin(dist, axis=0)
  dist2_vec = np.amin(dist, axis=1)
  # w1 = 1/(1+ np.argsort(dist1_vec))
  # w2 = 1/(1+ np.argsort(dist2_vec))
  # dist1 = np.average(dist1_vec, weights=w1)
  # dist2 = np.average(dist2_vec, weights=w2)
  dist1 = np.percentile(dist1_vec, 10)
  dist2 = np.percentile(dist2_vec, 10)
  # dist = distances(patch1, patch2)
  # dist1 = np.mean(np.sqrt(np.amin(dist, axis=0)))
  # dist2 = np.mean(np.sqrt(np.amin(dist, axis=1)))
  # dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  # dissim[p1['idx'], p2['idx']] = np.mean([dist1, dist2])
  # dissim[p2['idx'], p1['idx']] = np.mean([dist1, dist2])
  return p1['idx'], p2['idx'], np.mean(np.array([dist1, dist2]))


def calcPatchDistmulti4centroid(dou):
  # for dou in tqdm(doues):
  p1, p2 = dou
  patch1 = p1['patch']
  patch2 = p2['patch']
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist = np.sqrt(np.around(distances(patch1, patch2),3))
  dist1_vec = np.amin(dist, axis=0)
  dist2_vec = np.amin(dist, axis=1)
  # w1 = 1/(1+ np.argsort(dist1_vec))
  # w2 = 1/(1+ np.argsort(dist2_vec))
  # dist1 = np.average(dist1_vec, weights=w1)
  # dist2 = np.average(dist2_vec, weights=w2)
  dist1 = np.percentile(dist1_vec, 10)
  dist2 = np.percentile(dist2_vec, 10)
  # dist = distances(patch1, patch2)
  # dist1 = np.mean(np.sqrt(np.amin(dist, axis=0)))
  # dist2 = np.mean(np.sqrt(np.amin(dist, axis=1)))
  # dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  # dissim[p1['idx'], p2['idx']] = np.mean([dist1, dist2])
  # dissim[p2['idx'], p1['idx']] = np.mean([dist1, dist2])
  return p1['temp_idx'], p2['temp_idx'], np.mean(np.array([dist1, dist2]))



def process_single_patch(args):
    i, patches, keep_percent = args
    N = len(patches)
    row_results = []
    for j in range(N):
        if i == j:
            continue
        _, _, dist = calcPatchDistmulti((patches[i], patches[j]))
        row_results.append((j, dist))
    k = max(1, int(len(row_results) * keep_percent / 100))
    return i, nsmallest(k, row_results, key=lambda x: x[1])

def create_dissim_sparse(patches, keep_percent=1, n_jobs=16):
    """
    Create a sparse symmetric dissimilarity matrix based on custom patch distances.

    Parameters:
    - patches: list of dictionaries, each with 'patch' and 'idx'
    - keep_percent: percentage of nearest neighbors to retain (default: 5%)
    - n_jobs: number of processes for parallel computation (default: all available cores)

    Returns:
    - dissim_sparse: scipy.sparse.csr_matrix of shape (N, N)
    """
    N = len(patches)
    dissim_sparse = lil_matrix((N, N))

    args_list = [(i, patches, keep_percent) for i in range(N)]

    with mp.Pool(processes=n_jobs) as pool:
        for i, top_neighbors in tqdm(pool.imap_unordered(process_single_patch, args_list), total=N, desc="Processing rows"):
            for j, dist in top_neighbors:
                dissim_sparse[i, j] = dist
                dissim_sparse[j, i] = dist  # ensure symmetry

    return dissim_sparse.tocsr()


def calcPatchDistmulti_mean(dou):
  # for dou in tqdm(doues):
  p1, p2 = dou
  patch1 = p1['patch']
  patch2 = p2['patch']
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist = np.sqrt(np.around(distances(patch1, patch2),3))
  dist1_vec = np.amin(dist, axis=0)
  dist2_vec = np.amin(dist, axis=1)
  # sorted_v1 = dist1_vec[np.argsort(dist1_vec)]
  # sorted_v2 = dist2_vec[np.argsort(dist2_vec)]
  # w = np.linspace(1,0,len(sorted_v1))
  # w1 = 1/(1+ np.argsort(dist1_vec))
  # w2 = 1/(1+ np.argsort(dist2_vec))
  dist1 = np.mean(dist1_vec)
  dist2 = np.mean(dist2_vec)
  # dist1 = np.percentile(dist1_vec, 0)
  # dist2 = np.percentile(dist2_vec, 0)
  # dist = distances(patch1, patch2)
  # dist1 = np.mean(np.sqrt(np.amin(dist, axis=0)))
  # dist2 = np.mean(np.sqrt(np.amin(dist, axis=1)))
  # dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  # dissim[p1['idx'], p2['idx']] = np.mean([dist1, dist2])
  # dissim[p2['idx'], p1['idx']] = np.mean([dist1, dist2])
  return p1['idx'], p2['idx'], np.mean(np.array([dist1, dist2])), p1['seg']==p2['seg']

def calcPatchDistmulti_avg(dou):
  # for dou in tqdm(doues):
  p1, p2 = dou
  patch1 = p1['patch']
  patch2 = p2['patch']
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist = np.sqrt(np.around(distances(patch1, patch2),3))
  dist1_vec = np.amin(dist, axis=0)
  dist2_vec = np.amin(dist, axis=1)
  sorted_v1 = dist1_vec[np.argsort(dist1_vec)]
  sorted_v2 = dist2_vec[np.argsort(dist2_vec)]
  w = np.linspace(1,0,len(sorted_v1))
  # w1 = 1/(1+ np.argsort(dist1_vec))
  # w2 = 1/(1+ np.argsort(dist2_vec))
  dist1 = np.average(sorted_v1, weights=w)
  dist2 = np.average(sorted_v2, weights=w)
  # dist1 = np.percentile(dist1_vec, 0)
  # dist2 = np.percentile(dist2_vec, 0)
  # dist = distances(patch1, patch2)
  # dist1 = np.mean(np.sqrt(np.amin(dist, axis=0)))
  # dist2 = np.mean(np.sqrt(np.amin(dist, axis=1)))
  # dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  # dissim[p1['idx'], p2['idx']] = np.mean([dist1, dist2])
  # dissim[p2['idx'], p1['idx']] = np.mean([dist1, dist2])
  return p1['idx'], p2['idx'], np.mean(np.array([dist1, dist2])), p1['seg']==p2['seg']

def calcPatchDist4lda(dou):
  # for dou in tqdm(doues):
  p1, p2 = dou
  patch1 = p1['patch']
  patch2 = p2['patch']
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist = np.sqrt(np.around(distances(patch1, patch2),3))
  dist1_vec = np.amin(dist, axis=0)
  dist2_vec = np.amin(dist, axis=1)
  # w1 = 1/(1+ np.argsort(dist1_vec))
  # w2 = 1/(1+ np.argsort(dist2_vec))
  # dist1 = np.average(dist1_vec, weights=w1)
  # dist2 = np.average(dist2_vec, weights=w2)
  # dist1 = np.mean(dist1_vec)
  # dist2 = np.mean(dist2_vec)
  # dist = distances(patch1, patch2)
  # dist1 = np.mean(np.sqrt(np.amin(dist, axis=0)))
  # dist2 = np.mean(np.sqrt(np.amin(dist, axis=1)))
  # dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  # dissim[p1['idx'], p2['idx']] = np.mean([dist1, dist2])
  # dissim[p2['idx'], p1['idx']] = np.mean([dist1, dist2])
  return p1['idx'], p2['idx'], sorted(np.hstack((dist1_vec, dist2_vec))), p1['seg']==p2['seg']

# @cuda.jit
def calcPatchDisttorch(patch1, patch2):
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2]).to('cuda')
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2]).to('cuda')
  dist1 = torch.mean(torch.Tensor([torch.amin(torch.Tensor([torch.linalg.norm(pix2-pix1,2) for pix1 in patch1])) for pix2 in patch2]))
  dist2 = torch.mean(torch.Tensor([torch.amin(torch.Tensor([torch.linalg.norm(pix2-pix1,2) for pix2 in patch2])) for pix1 in patch1]))
  return torch.mean(torch.Tensor([dist1, dist2]))
  patch1 = p1['patch']
  patch2 = p2['patch']
  if len(patch1.shape) > 2:
    patch1 = patch1.reshape(patch1.shape[0] * patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist1 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix1 in patch1]) for pix2 in patch2])
  dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])


# @overload(np.amin)
# def jit_min(dist, axis):
#   dmin = []
#   if axis==0:
#     for i in range(dist.shape[axis]):
#       d = np.Inf
#       for k in range(dist.shape[1-axis]):
#         if dist[i, k] < d:
#           d = dist[i,k]
#       dmin.append(d)
#     return np.array(dmin).mean()
#   if axis ==1:
#     for i in range(dist.shape[axis]):
#       d = np.Inf
#       for k in range(dist.shape[1-axis]):
#         if dist[k, i] < d:
#           d = dist[k, i]
#       dmin.append(d)
#     return np.array(dmin).mean()
#
#
# @overload(np.linalg.norm)
# def oneD_norm_2(a):
#   val = np.abs(a)
#   return np.sqrt(np.sum(val*val))
#
#
# @cuda.jit()
# def calcDistGPUPar(patches, dissim):
#   i, j = cuda.grid(2)
#   if i<4224 and j<4224:
#     for patch1 in patches:
#       for patch2 in patches:
#         if len(patch1.shape) > 2:
#           p1 = patch1.copy()
#           p1 = p1.reshape(patch1.shape[0] * patch1.shape[1], patch1.shape[2])
#           p2 = patch2.copy()
#           p2 = p2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
#         dist = np.zeros((p1.shape[0], p2.shape[0]))
#         for l, pix1 in enumerate(p1):
#           for k, pix2 in enumerate(p2):
#             dist[l,k] = np.linalg.norm(pix2 - pix1)
#         dist = np.array([np.amin(np.array([np.linalg.norm(pix2 - pix1) for pix1 in p1])) for pix2 in p2])
#   #       # dist2 = np.array([np.array([np.linalg.norm(pix2 - pix1, 2) for pix2 in p2]) for pix1 in p1])
#   #       dmin0 = []
#   #       for i in range(dist.shape[0]):
#   #         d = np.Inf
#   #         for k in range(dist.shape[1]):
#   #           if dist[i, k] < d:
#   #             d = dist[i, k]
#   #         dmin0.append(d)
#   #       dmin1 = []
#   #       for i in range(dist.shape[1]):
#   #         d = np.Inf
#   #         for k in range(dist.shape[0]):
#   #           if dist[k, i] < d:
#   #             d = dist[k, 1]
#   #         dmin1.append(d)
#   #       d = np.array([np.array(dmin1).mean(), np.array(dmin0).mean()]).mean()
#   #       dissim[i, j] = d
#   #       dissim[j, i] = d
#   # return dissim

def getPatchDS(sess, image, GT, fuse, patch_size=4, ifPCA=True, n=16, filenames=None):
  pos_dist = []
  neg_dist = []
  for i in range(25):
    im, gt, rep = sess.run([image, GT, fuse])
    rep = np.squeeze(rep)
    if ifPCA:
      res_vec = rep.reshape((rep.shape[0] * rep.shape[1], rep.shape[2]))
      res_reduced = PCA(n_components=n, svd_solver='full').fit_transform(res_vec)
      rep = np.reshape(res_reduced, (rep.shape[0], rep.shape[1], n))
      # rep_3 = rep_3 - np.amin(rep_3) / np.amax(rep_3 - np.amin(rep_3))
    xVec = np.arange(0, rep.shape[0] - 1, 10)
    yVec = np.arange(0, rep.shape[1] - 1, 10)
    center = int(rep.shape[0]/2), int(rep.shape[1]/2)
    for x in xVec:
      for y in yVec:
        x_pos = x + np.random.randint(-1, 1) * int(patch_size/2)
        y_pos = y + np.random.randint(-1, 1) * int(patch_size/2)
        if x_pos<0 or y_pos<0:
          x_pos=x
          y_pos=y
        if x_pos==x and y_pos==y:
          x_pos = x+int(patch_size/2)
        if x < center[0]:
          x_neg = x+ center[0]
        else:
          x_neg = x - center[0]
        if y < center[1]:
          y_neg = y+ center[1]
        else:
          y_neg = y - center[1]
        patch_orig = rep[x:x+patch_size, y:y+patch_size, :]
        patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]

        pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return pos_dist, neg_dist

def getPatchDS_new(sess, image, fuse, patch_size=8, ifPCA=True, n=16, filenames=None, perClass=False):
  patches = []
  ims = []
  reps = []
  gts = []
  for i in range(100):
    im, rep = sess.run([image, fuse])
    rep = np.squeeze(rep)
    if perClass:
      gt_name = 'BSDS500/data/groundTruth/test/{}.png'.format(filenames[i])
      gt = plt.imread(gt_name)
      try:
        if len(np.unique(gt))<2:
          continue
      except:
          print(np.unique(gt))
    else:
      gt_name = 'BSDS500/data/groundTruth/test/{}.npy'.format(filenames[i])
      gt = np.load(gt_name, allow_pickle=True).item()['groundTruth'][0][0][0][0][0]
    ims.append(im)
    reps.append(rep)
    gts.append(gt)
  if ifPCA:
    reps = pcarep(reps, n)
    # res_vec = rep.reshape((rep.shape[0] * rep.shape[1], rep.shape[2]))
    # res_reduced = PCA(n_components=n, svd_solver='full').fit_transform(res_vec)
    # rep = np.reshape(res_reduced, (rep.shape[0], rep.shape[1], n))
    # rep = (rep - np.amin(rep)) / (np.amax(rep)- np.amin(rep))
  for i in range(25):
    im = ims[i]
    rep = reps[i]
    gt = gts[i]
    xVec = np.arange(0, rep.shape[0] - 1, 10)
    yVec = np.arange(0, rep.shape[1] - 1, 10)
    center = int(rep.shape[0]/2), int(rep.shape[1]/2)
    countbg = 0
    for x in xVec:
      for y in yVec:
        # x_pos = x + np.random.randint(-1, 1) * int(patch_size/2)
        # y_pos = y + np.random.randint(-1, 1) * int(patch_size/2)
        # if x_pos<0 or y_pos<0:
        #   x_pos=x
        #   y_pos=y
        # if x_pos==x and y_pos==y:
        #   x_pos = x+int(patch_size/2)
        # if x < center[0]:
        #   x_neg = x+ center[0]
        # else:
        #   x_neg = x - center[0]
        # if y < center[1]:
        #   y_neg = y+ center[1]
        # else:
        #   y_neg = y - center[1]
        patch_orig = rep[x:x+patch_size, y:y+patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        seg_orig = gt[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4]
        try:
          if mostCommonClass(seg_orig[:,:,0]) is not None:
            print(mostCommonClass(seg_orig[:,:,0]))
          else:
            continue
        except:
          if mostCommonClass(seg_orig) is not None:
            print(mostCommonClass(seg_orig))
          else:
            continue
        fname = filenames[i]
        if perClass:
          seg = 1 if mostCommonClass(seg_orig[:,:,0])==1.0 else 0
          if seg==1:
            if countbg >3:
              continue
            countbg += 1
          label = int(fname)//10
          patches.append(
            {"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": seg, "label": label})
        else:
          patches.append(
            {"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig), "im": fname})
  return patches


def getPatchDS_rect(sess, image, fuse, patch_size=8, ifPCA=True, n=3, filenames=None):
  # patches = []
  # fig, axs = plt.subplots(3, 4)
  # fig.tight_layout()
  # axs.flatten()
  dir = 'rectangleExp/bsds'
  reps = []
  for i in range(12):
    im, rep = sess.run([image, fuse])
    reps.append(np.squeeze(rep))

  if ifPCA:
    reps = np.array(reps)
    res_red = reps.reshape((reps.shape[0] * reps.shape[1] * reps.shape[2], reps.shape[3]))
    res_reduced = PCA(n_components=n, svd_solver='full').fit_transform(res_red)
    reps = np.reshape(res_reduced, (reps.shape[0], reps.shape[1], reps.shape[2], n))
    reps = (reps - np.amin(reps)) / np.amax(reps - np.amin(reps))
    for i in range(12):
      plt.imsave(os.path.join(dir, 'rep_' + str(i) + '.jpg'), reps[i])

  # plt.show()

  # return patches

def getPatchDS_imagenette(sess, image, labels, fuse, patch_size=4, ifPCA=True, n=16):
  # pos_dist = []
  # neg_dist = []
  patches = []
  for i in range(20):
    im, label, rep = sess.run([image, labels, fuse])
    rep = np.squeeze(rep)
    if ifPCA:
      res_vec = rep.reshape((rep.shape[0] * rep.shape[1], rep.shape[2]))
      res_reduced = PCA(n_components=n, svd_solver='full').fit_transform(res_vec)
      rep = np.reshape(res_reduced, (rep.shape[0], rep.shape[1], n))
      # rep_3 = rep_3 - np.amin(rep_3) / np.amax(rep_3 - np.amin(rep_3))
    xVec = np.arange(0, rep.shape[0] - 1, 20)
    yVec = np.arange(0, rep.shape[1] - 1, 20)
    center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
    for x in xVec:
      for y in yVec:
        x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        if x_pos == x and y_pos == y:
          x_pos = x + int(patch_size / 2)
        if x_pos < 0:
          x_pos = x + int(patch_size / 2)
        if y_pos < 0:
          y_pos = y + int(patch_size / 2)
        if x_pos+patch_size > rep.shape[0]:
          x_pos = x-int(patch_size / 2)
        if y_pos+patch_size > rep.shape[1]:
          y_pos = y-int(patch_size / 2)
        if x < center[0]:
          x_neg = x + (center[0]-patch_size)
        else:
          x_neg = x - (center[0]-patch_size)
        if y < center[1]:
          y_neg = y + (center[1]-patch_size)
        else:
          y_neg = y - (center[1]-patch_size)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "label": label})
        patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "label": label})
        patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches


def getPatchDS_COCO(sess, image,mask, labels, fuse, patch_size=4, ifPCA=True, n=16):
  # pos_dist = []
  # neg_dist = []
  dataDir = './COCO'
  annFile = '{}/annotations/instances_train2017.json'.format(dataDir)
  coco =COCO(annFile)
  catIDs = coco.getCatIds()
  cats = coco.loadCats(catIDs)
  filterClasses = ['dog', 'bear', 'airplane', 'train', 'banana']
  # filterClasses = ['dog', 'elephant', 'fire hydrant', 'train', 'airplane']
  catIds = coco.getCatIds(catNms=filterClasses)
  patches = []
  for i in range(22):
    im, seg, label, rep = sess.run([image, mask, labels, fuse])
    rep = np.squeeze(rep)
    if ifPCA:
      res_vec = rep.reshape((rep.shape[0] * rep.shape[1], rep.shape[2]))
      res_reduced = PCA(n_components=n, svd_solver='full').fit_transform(res_vec)
      rep = np.reshape(res_reduced, (rep.shape[0], rep.shape[1], n))
      # rep_3 = rep_3 - np.amin(rep_3) / np.amax(rep_3 - np.amin(rep_3))
    xVec = np.arange(0, rep.shape[0] - 1, 10)
    yVec = np.arange(0, rep.shape[1] - 1, 10)
    center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
    for x in xVec:
      for y in yVec:
        x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        if x_pos == x and y_pos == y:
          x_pos = x + int(patch_size / 2)
        if x_pos < 0:
          x_pos = x + int(patch_size / 2)
        if y_pos < 0:
          y_pos = y + int(patch_size / 2)
        if x_pos+patch_size > rep.shape[0]:
          x_pos = x-int(patch_size / 2)
        if y_pos+patch_size > rep.shape[1]:
          y_pos = y-int(patch_size / 2)
        if x < center[0]:
          x_neg = x + (center[0]-patch_size)
        else:
          x_neg = x - (center[0]-patch_size)
        if y < center[1]:
          y_neg = y + (center[1]-patch_size)
        else:
          y_neg = y - (center[1]-patch_size)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        seg_orig = seg[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        seg_pos = seg[x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        seg_neg = seg[x_neg * 4: x_neg * 4 + patch_size * 4, y_neg * 4:y_neg * 4 + patch_size * 4, :]
        patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig), "label": label})
        patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "seg": mostCommonClass(seg_pos), "label": label})
        patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "seg": mostCommonClass(seg_neg), "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches

def pcarep(allreps, n):
  reps = random.sample(allreps, 100)
  allreps_1 = np.array(allreps[:int(len(allreps)/2)])
  reps = np.array(reps)
  res_red = reps.reshape((reps.shape[0] * reps.shape[1] * reps.shape[2], reps.shape[3]))
  pca_fit = PCA(n_components=n).fit(res_red)
  allreps_new_1 = allreps_1.reshape((allreps_1.shape[0] * allreps_1.shape[1] * allreps_1.shape[2], allreps_1.shape[3]))
  res_reduced_1 = pca_fit.transform(allreps_new_1)
  allreps_1 = np.array(allreps[int(len(allreps)/2):])
  allreps_new_1 = allreps_1.reshape((allreps_1.shape[0] * allreps_1.shape[1] * allreps_1.shape[2], allreps_1.shape[3]))
  res_reduced_2 = pca_fit.transform(allreps_new_1)
  res_reduced = np.concatenate([res_reduced_1, res_reduced_2])
  reps = np.reshape(res_reduced, (len(allreps), allreps[0].shape[0], allreps[0].shape[1], n))
  reps = (reps - np.amin(reps)) / np.amax(reps - np.amin(reps))
  return reps

def pcarep_new(allreps, n):
  reps = random.sample(allreps, 150)
  batches = np.array_split(allreps, 5)
  reps = np.array(reps)
  res_red = reps.reshape((reps.shape[0] * reps.shape[1] * reps.shape[2], reps.shape[3]))
  pca_fit = PCA(n_components=n).fit(res_red)
  res_red_all = []
  for b in batches:
    allreps_new = b.reshape((b.shape[0] * b.shape[1] * b.shape[2], b.shape[3]))
    res_red_all.append(pca_fit.transform(allreps_new))
    # allreps = np.array(allreps[int(len(allreps)/2):])
    # allreps_new = allreps_1.reshape((allreps_1.shape[0] * allreps_1.shape[1] * allreps_1.shape[2], allreps_1.shape[3]))
    # res_reduced_2 = pca_fit.transform(allreps_new_1)
  res_reduced = np.concatenate(res_red_all)
  reps = np.reshape(res_reduced, (len(allreps), allreps[0].shape[0], allreps[0].shape[1], n))
  reps = (reps - np.amin(reps)) / np.amax(reps - np.amin(reps))
  return reps


def pcafit(allreps, n):
  reps = random.sample(allreps, 200)
  # batches = np.array_split(allreps, 5)
  reps = np.array(reps)
  res_red = reps.reshape((reps.shape[0] * reps.shape[1] * reps.shape[2], reps.shape[3]))
  pca_fit = PCA(n_components=n).fit(res_red)
  return pca_fit


def pcarep_MIT(allreps, n, pca_fit=None):
  if not pca_fit:
    reps = random.sample(allreps, 150)
    reps = np.array(reps)
    res_red = reps.reshape((reps.shape[0] * reps.shape[1] * reps.shape[2], reps.shape[3]))
    pca_fit = PCA(n_components=n).fit(res_red)
  res_red_all = []
  # batches = np.array_split(allreps, 4)
  batches = [allreps[:int(len(allreps)/4)], allreps[int(len(allreps)/4):2*int(len(allreps)/4)], allreps[2*int(len(allreps)/4):3*int(len(allreps)/4)], allreps[3*int(len(allreps)/4):]]
  for b in batches:
    b = np.array(b)
    allreps_new = b.reshape((b.shape[0] * b.shape[1] * b.shape[2], b.shape[3]))
    res_red_all.append(pca_fit.transform(allreps_new))
    # allreps = np.array(allreps[int(len(allreps)/2):])
    # allreps_new = allreps_1.reshape((allreps_1.shape[0] * allreps_1.shape[1] * allreps_1.shape[2], allreps_1.shape[3]))
    # res_reduced_2 = pca_fit.transform(allreps_new_1)
  res_reduced = np.concatenate(res_red_all)
  reps = np.reshape(res_reduced, (len(allreps), allreps[0].shape[0], allreps[0].shape[1], n))
  reps = (reps - np.amin(reps)) / np.amax(reps - np.amin(reps))
  return reps, pca_fit

def Ipcarep(allreps, n):
  allreps = np.array(allreps)
  reps = random.sample(allreps, 200)
  res_red = reps.reshape((reps.shape[0] * reps.shape[1] * reps.shape[2], reps.shape[3]))
  IncrementalPCA(n_components=n).fit(res_red)
  allreps = allreps.reshape((allreps.shape[0] * allreps.shape[1] * allreps.shape[2], allreps.shape[3]))
  res_reduced = IncrementalPCA(n_components=n).transform(allreps)
  reps = np.reshape(res_reduced, (allreps.shape[0], allreps.shape[1], allreps.shape[2], n))
  reps = (reps - np.amin(reps)) / np.amax(reps - np.amin(reps))
  return reps

def getPatchDS_frompkl(patch_size=8):
  with open('DS.pkl', 'rb') as f:
    ims, segs, labs, reps = pickle.load(f)
  reps = pcarep(reps, n=16)
  patches = []
  for i in tqdm(range(449)):
    im = ims[i]
    seg = segs[i]
    label =labs[i]
    rep = reps[i]
    xVec = np.arange(0, rep.shape[0] - 1, 8)
    yVec = np.arange(0, rep.shape[1] - 1, 8)
    center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
    for x in xVec:
      for y in yVec:
        # x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        # y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        # if x_pos == x and y_pos == y:
        #   x_pos = x + int(patch_size / 2)
        # if x_pos < 0:
        #   x_pos = x + int(patch_size / 2)
        # if y_pos < 0:
        #   y_pos = y + int(patch_size / 2)
        # if x_pos+patch_size > rep.shape[0]:
        #   x_pos = x-int(patch_size / 2)
        # if y_pos+patch_size > rep.shape[1]:
        #   y_pos = y-int(patch_size / 2)
        # if x < center[0]:
        #   x_neg = x + (center[0]-patch_size)
        # else:
        #   x_neg = x - (center[0]-patch_size)
        # if y < center[1]:
        #   y_neg = y + (center[1]-patch_size)
        # else:
        #   y_neg = y - (center[1]-patch_size)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        seg_orig = seg[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        # patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        # im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # seg_pos = seg[x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        # im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        # seg_neg = seg[x_neg * 4: x_neg * 4 + patch_size * 4, y_neg * 4:y_neg * 4 + patch_size * 4, :]
        try:
          mostCommonClass(seg_orig)
        except:
          print(seg_orig)
        patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig), "label": label})
        # patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "seg": mostCommonClass(seg_pos), "label": label})
        # patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "seg": mostCommonClass(seg_neg), "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches


def getPatchDS_pascal(sess, image, mask, imname, fuse, patch_size=4, ifPCA=True, n=16, BG=True):
  patches = []
  ims = []
  segs = []
  reps = []
  imnames = []
  # counter = {}
  # for i in tqdm(range(449)):
  PASCAL_VOC_classes = {str([68,1,84,255]): 'background', str([68,2,85,255]): 'airplane', str([68,3,87,255]): 'bicycle', str([69,5,88,255]): 'bird',
                        str([69,6,90,255]): 'boat',
                        str([69,8,91,255]): 'bottle', str([70,9,92,255]): 'bus', str([70,11,94,255]): 'car', str([70,12,95,255]): 'cat', str([70,14,97,255]): 'chair',
                        str([71,15,98,255]): 'cow',
                        str([71,17,99,255]): 'table', str([71,18,101,255]): 'dog', str([71,20,102,255]): 'horse', str([71,21,103,255]): 'motorbike',
                        str([71,22,105,255]): 'person', str([71,24,106,255]): 'potted_plant', str([72,25,107,255]): 'sheep', str([72,26,108,255]): 'sofa', str([72,28,110,255]): 'train',
                        str([72,29,111,255]): 'tv',
                        str([253,231,36,255]): 'void'}

  # for i in tqdm(range(200)):
  #   im, seg, imn, rep = sess.run([image, mask, imname, fuse])
  #   reps.append(np.squeeze(rep))
  #   ims.append(im)
  #   # mn = seg[:3].min()
  #   # mx = seg[:3].max()
  #   # mx -= mn
  #   # seg = (((seg - mn) / mx) * 255).astype(np.uint8)
  #   # seg[seg==30]=0
  #   # seg[seg==215]=255
  #   segs.append(seg)
  #   imnames.append(imn)
  #   # labelsInim = np.unique(seg.reshape(-1, 4), axis=0, return_counts=True)
  #   # for l in labelsInim[0]:
  #   #   if str(l).replace(" ", "") not in [str(item).replace(' ','').replace(',','') for item in PASCAL_VOC_classes.keys()]:
  #   #     continue


  if ifPCA:
    # reps = pcarep_new(reps, n)
    # pcaFit = pcafit(reps, n)

    # with open('pcafit.pkl', 'wb') as f:
    #   pickle.dump(pcaFit, f)

    with open('pcafit.pkl', 'rb') as f:
      pcaFit = pickle.load(f)
    # for rep in tqdm(reps):

    # rep_forplot = pcarep(reps, 3)
    # rep_forplot = (rep_forplot - np.amin(rep_forplot)) / np.amax(rep_forplot - np.amin(rep_forplot))

  # for i in tqdm(range(449)):
  # idx_nobg = 0
  counter = {}

  for i in tqdm(range(2913)):
    im, seg, imn, rep = sess.run([image, mask, imname, fuse])
    # im = ims[i]
    # seg = segs[i]
    # rep = reps[i]
    # imn = imnames[i]
    # flag = 0
    temp_rep = np.squeeze(rep).reshape((rep.shape[1] * rep.shape[2], rep.shape[3]))
    rep = pcaFit.transform(temp_rep).reshape((rep.shape[1], rep.shape[2], n))
    xVec = np.arange(0, rep.shape[0] - 1, 10)
    yVec = np.arange(0, rep.shape[1] - 1, 10)
    center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
    for x in xVec:
      for y in yVec:
        # x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        # y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        # if x_pos == x and y_pos == y:
        #   x_pos = x + int(patch_size / 2)
        # if x_pos < 0:
        #   x_pos = x + int(patch_size / 2)
        # if y_pos < 0:
        #   y_pos = y + int(patch_size / 2)
        # if x_pos+patch_size > rep.shape[0]:
        #   x_pos = x-int(patch_size / 2)
        # if y_pos+patch_size > rep.shape[1]:
        #   y_pos = y-int(patch_size / 2)
        # if x < center[0]:
        #   x_neg = x + (center[0]-patch_size)
        # else:
        #   x_neg = x - (center[0]-patch_size)
        # if y < center[1]:
        #   y_neg = y + (center[1]-patch_size)
        # else:
        #   y_neg = y - (center[1]-patch_size)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        seg_orig = seg[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        # patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        # im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # seg_pos = seg[x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        # im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        # seg_neg = seg[x_neg * 4: x_neg * 4 + patch_size * 4, y_neg * 4:y_neg * 4 + patch_size * 4, :]
        mostcom_rgba = mostCommonClass_pascal(seg_orig)
        if mostcom_rgba is not None:
          mostcom = PASCAL_VOC_classes[str(list(mostcom_rgba))]
        else:
          mostcom = None
        if mostcom=='void' or mostcom=='bicycle' or mostcom=='sofa':
          continue
        elif mostcom=='background' and not BG:
            continue

        elif mostcom is None:
          continue
        elif mostcom in counter.keys():
          # if counter[mostcom]>=300:
          #   continue
          # else:
          counter[mostcom] +=1
        else:
            counter[mostcom]=1

        patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostcom, "im_id": i, "im_ref": imn})
        # else:
        #   patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig)})
        # patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "seg": mostCommonClass(seg_pos), "label": label})
        # patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "seg": mostCommonClass(seg_neg), "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches


def getPatchDS_MIT(sess, image, label, imname, fuse, patch_size=8, ifPCA=True, n=16):
  patches = []
  ims = []
  segs = []
  reps = []
  imnames = []
  # for i in tqdm(range(449)):
  for _ in tqdm(range(800)):
    im, seg, imn, rep = sess.run([image, label, imname, fuse])
    reps.append(np.squeeze(rep))
    ims.append(im)
    # mn = seg[:3].min()
    # mx = seg[:3].max()
    # mx -= mn
    # seg = (((seg - mn) / mx) * 255).astype(np.uint8)
    # seg[seg==30]=0
    # seg[seg==215]=255
    segs.append(seg)
    imnames.append(imn)

  if ifPCA:
    reps_pca, pca_fit = pcarep_MIT(reps, n)

  ims2 = []
  segs2 = []
  reps2 = []
  imnames2 = []
  for _ in tqdm(range(600)):
    im, seg, imn, rep = sess.run([image, label, imname, fuse])
    reps2.append(np.squeeze(rep))
    ims2.append(im)
    # mn = seg[:3].min()
    # mx = seg[:3].max()
    # mx -= mn
    # seg = (((seg - mn) / mx) * 255).astype(np.uint8)
    # seg[seg==30]=0
    # seg[seg==215]=255
    segs2.append(seg)
    imnames2.append(imn)

  if ifPCA:
    reps_pca2, _ = pcarep_MIT(reps2, n, pca_fit)
    # rep_forplot = pcarep(reps, 3)
    # rep_forplot = (rep_forplot - np.amin(rep_forplot)) / np.amax(rep_forplot - np.amin(rep_forplot))

  reps = np.concatenate([reps_pca, reps_pca2])
  ims.extend(ims2)
  segs.extend(segs2)
  imnames.extend(imnames2)
  # for i in tqdm(range(449)):
  # idx_nobg = 0
  counter = {}
  for i in tqdm(range(1400)):
    im = ims[i]
    seg = segs[i]
    rep = reps[i]
    imn = imnames[i]
    flag = 0
    xVec = np.arange(4, rep.shape[0] - 1, 10)
    yVec = np.arange(4, rep.shape[1] - 1, 10)
    center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
    for x in xVec:
      for y in yVec:
        # x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        # y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        # if x_pos == x and y_pos == y:
        #   x_pos = x + int(patch_size / 2)
        # if x_pos < 0:
        #   x_pos = x + int(patch_size / 2)
        # if y_pos < 0:
        #   y_pos = y + int(patch_size / 2)
        # if x_pos+patch_size > rep.shape[0]:
        #   x_pos = x-int(patch_size / 2)
        # if y_pos+patch_size > rep.shape[1]:
        #   y_pos = y-int(patch_size / 2)
        # if x < center[0]:
        #   x_neg = x + (center[0]-patch_size)
        # else:
        #   x_neg = x - (center[0]-patch_size)
        # if y < center[1]:
        #   y_neg = y + (center[1]-patch_size)
        # else:
        #   y_neg = y - (center[1]-patch_size)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        # seg_orig = seg[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        # patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        # im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # seg_pos = seg[x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        # im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        # seg_neg = seg[x_neg * 4: x_neg * 4 + patch_size * 4, y_neg * 4:y_neg * 4 + patch_size * 4, :]
        # mostcom = mostCommonClass(seg_orig)
        # if mostcom==215:
        #   continue
        # elif mostcom==30:
        #   if BG:
        #     if flag>6:
        #       continue
        #     flag+=1
        #   else:
        #     continue
        # elif mostcom is None:
        #   continue
        # elif mostcom in counter.keys():
        #   if counter[mostcom]>=300:
        #     continue
        #   else:
        #     counter[mostcom] +=1
        # else:
        #     counter[mostcom]=1
        if seg in counter.keys():
          if counter[seg]>=2000:
            continue
          else:
            counter[seg] += 1
        else:
          counter[seg] = 1
        patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": seg, "im_id": i, "im_ref": imn})
        # else:
        #   patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig)})
        # patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "seg": mostCommonClass(seg_pos), "label": label})
        # patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "seg": mostCommonClass(seg_neg), "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches



def getPatchDS_MINC(sess, image, label, fuse, patch_size=4, ifPCA=True, n=16):
  patches = []
  ims = []
  segs = []
  reps_orig = []

  # for i in tqdm(range(449)):
  for i in tqdm(range(150)):
    im, seg, rep = sess.run([image, label, fuse])
    reps_orig.append(np.squeeze(rep))
    ims.append(im)
    # mn = seg[:3].min()
    # mx = seg[:3].max()
    # mx -= mn
    # seg = (((seg - mn) / mx) * 255).astype(np.uint8)
    # seg[seg==30]=0
    # seg[seg==215]=255
    segs.append([seg[0].decode('utf-8'), seg[1].decode('utf-8')])

  if ifPCA:
    reps = pcarep(reps_orig, n)
    reps_norm = (reps - np.amin(reps)) / np.amax(reps - np.amin(reps))
    #
    reps_forplots = pcarep(reps_orig, 3)
    reps_forplots = (reps_forplots - np.amin(reps_forplots)) / np.amax(reps_forplots - np.amin(reps_forplots))

  # for i in tqdm(range(449)):
  for i in tqdm(range(150)):
    im = ims[i]
    seg = segs[i]
    rep = reps[i]
    rep_p = reps_forplots[i]
    rep_n =reps_norm[i]
    flag = 0
    xVec = np.arange(0, rep.shape[0] - 1, 20)
    yVec = np.arange(0, rep.shape[1] - 1, 20)
    center = [100, 300]
    for x in xVec:
      for y in yVec:
        # x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        # y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        # if x_pos == x and y_pos == y:
        #   x_pos = x + int(patch_size / 2)
        # if x_pos < 0:
        #   x_pos = x + int(patch_size / 2)
        # if y_pos < 0:
        #   y_pos = y + int(patch_size / 2)
        # if x_pos+patch_size > rep.shape[0]:
        #   x_pos = x-int(patch_size / 2)
        # if y_pos+patch_size > rep.shape[1]:
        #   y_pos = y-int(patch_size / 2)
        # if x < center[0]:
        #   x_neg = x + (center[0]-patch_size)
        # else:
        #   x_neg = x - (center[0]-patch_size)
        # if y < center[1]:
        #   y_neg = y + (center[1]-patch_size)
        # else:
        #   y_neg = y - (center[1]-patch_size)
        if (x * 4 <center[0]) and (x*4 + patch_size*4 > center[0]) or (x * 4 <center[1]) and (x*4 + patch_size*4 > center[1])\
                or (y * 4 <center[0]) and (y*4 + patch_size*4 > center[0]) or (y * 4 <center[1]) and (y*4 + patch_size*4 > center[1]):
          continue
        elif (x*4 + patch_size*4 < center[0]) or (x*4 > center[1]) or (y*4 + patch_size*4 < center[0])\
                or (y*4 > center[1]):
          lbl = seg[1]
        else:
          lbl = seg[0]
        print(x*4, y*4)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        patch_p = rep_p[x:x + patch_size, y:y + patch_size, :]
        patch_norm = rep_n[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        # seg_orig = seg[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        # patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        # im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # seg_pos = seg[x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        # im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        # seg_neg = seg[x_neg * 4: x_neg * 4 + patch_size * 4, y_neg * 4:y_neg * 4 + patch_size * 4, :]
        # mostcom = mostCommonClass(seg_orig)
        # if mostcom==215:
        #   continue
        # elif mostcom==30:
        #   if flag>3:
        #     continue
        #   flag+=1
        # elif mostcom is None:
        #   continue
        patches.append({"idx": len(patches), "patch": patch_orig, "patch_plot": patch_p, "patch_norm": patch_norm, "ref": im_orig, "seg": lbl, "im": i})
        # else:
        #   patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig)})
        # patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "seg": mostCommonClass(seg_pos), "label": label})
        # patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "seg": mostCommonClass(seg_neg), "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches


def getPatchDS_COCO_new(sess, image,mask, labels, fuse, patch_size=4, ifPCA=True, n=16):
  # pos_dist = []
  # neg_dist = []
  # dataDir = './COCO'
  # annFile = '{}/annotations/instances_train2017.json'.format(dataDir)
  # # annFile = 'cocoanns.json'
  # coco =COCO(annFile)
  # catIDs = coco.getCatIds()
  # cats = coco.loadCats(catIDs)
  # filterClasses = ['dog', 'bear', 'airplane', 'train', 'banana']
  # # filterClasses = ['bus', 'dog', 'pizza', 'scissors', 'airplane']
  # catIds = coco.getCatIds(catNms=filterClasses)
  patches = []
  ims = []
  segs = []
  labs = []
  reps = []

  # for i in tqdm(range(449)):
  for i in range(72):
    im, seg, label, rep = sess.run([image, mask, labels, fuse])
    reps.append(np.squeeze(rep))
    ims.append(im)
    labs.append(label)
    segs.append(seg)

  if ifPCA:
    reps = pcarep(reps, n)
    #
    # if n==3:
    #   rep = (rep - np.amin(rep)) / np.amax(rep - np.amin(rep))

  # for i in tqdm(range(449)):
  for i in tqdm(range(72)):
    im = ims[i]
    seg = segs[i]
    label =labs[i]
    rep = reps[i]
    xVec = np.arange(0, rep.shape[0] - 1, 10)
    yVec = np.arange(0, rep.shape[1] - 1, 10)
    center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
    for x in xVec:
      for y in yVec:
        # x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        # y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        # if x_pos == x and y_pos == y:
        #   x_pos = x + int(patch_size / 2)
        # if x_pos < 0:
        #   x_pos = x + int(patch_size / 2)
        # if y_pos < 0:
        #   y_pos = y + int(patch_size / 2)
        # if x_pos+patch_size > rep.shape[0]:
        #   x_pos = x-int(patch_size / 2)
        # if y_pos+patch_size > rep.shape[1]:
        #   y_pos = y-int(patch_size / 2)
        # if x < center[0]:
        #   x_neg = x + (center[0]-patch_size)
        # else:
        #   x_neg = x - (center[0]-patch_size)
        # if y < center[1]:
        #   y_neg = y + (center[1]-patch_size)
        # else:
        #   y_neg = y - (center[1]-patch_size)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        seg_orig = seg[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        # patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        # im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # seg_pos = seg[x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        # patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        # im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        # seg_neg = seg[x_neg * 4: x_neg * 4 + patch_size * 4, y_neg * 4:y_neg * 4 + patch_size * 4, :]
        try:
          mostCommonClass(seg_orig)
        except:
          print(seg_orig)
        patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig), "label": label})
        # patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "seg": mostCommonClass(seg_pos), "label": label})
        # patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "seg": mostCommonClass(seg_neg), "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches


# def getPatchDS_MINC(sess, image, label, fuse, patch_size=4, ifPCA=True, n=16):
#   patches = []
#   ims = []
#   labels = []
#   reps = []
#
#   # for i in tqdm(range(449)):
#   for i in range(230):
#     im, lbl, rep = sess.run([image, label, fuse])
#     reps.append(np.squeeze(rep))
#     ims.append(im)
#     # mn = seg[:3].min()
#     # mx = seg[:3].max()
#     # mx -= mn
#     # seg = (((seg - mn) / mx) * 255).astype(np.uint8)
#     # seg[seg==30]=0
#     # seg[seg==215]=255
#     labels.append(lbl)
#
#   if ifPCA:
#     reps = pcarep(reps, n)
#     #
#     # if n==3:
#     #   rep = (rep - np.amin(rep)) / np.amax(rep - np.amin(rep))
#
#   # for i in tqdm(range(449)):
#   for i in tqdm(range(230)):
#     im = ims[i]
#     lbl = labels[i]
#     rep = reps[i]
#     flag = 0
#     xVec = np.arange(0, rep.shape[0] - 1, 10)
#     yVec = np.arange(0, rep.shape[1] - 1, 10)
#     center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
#     for x in xVec:
#       for y in yVec:
#         # x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
#         # y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
#         # if x_pos == x and y_pos == y:
#         #   x_pos = x + int(patch_size / 2)
#         # if x_pos < 0:
#         #   x_pos = x + int(patch_size / 2)
#         # if y_pos < 0:
#         #   y_pos = y + int(patch_size / 2)
#         # if x_pos+patch_size > rep.shape[0]:
#         #   x_pos = x-int(patch_size / 2)
#         # if y_pos+patch_size > rep.shape[1]:
#         #   y_pos = y-int(patch_size / 2)
#         # if x < center[0]:
#         #   x_neg = x + (center[0]-patch_size)
#         # else:
#         #   x_neg = x - (center[0]-patch_size)
#         # if y < center[1]:
#         #   y_neg = y + (center[1]-patch_size)
#         # else:
#         #   y_neg = y - (center[1]-patch_size)
#         patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
#         im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
#         seg_orig = seg[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
#         # patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
#         # im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
#         # seg_pos = seg[x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
#         # patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
#         # im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
#         # seg_neg = seg[x_neg * 4: x_neg * 4 + patch_size * 4, y_neg * 4:y_neg * 4 + patch_size * 4, :]
#         mostcom = mostCommonClass(seg_orig)
#         if mostcom==215:
#           continue
#         elif mostcom==30:
#           if flag>3:
#             continue
#           flag+=1
#         elif mostcom is None:
#           continue
#         patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig), "im": i})
#         # else:
#         #   patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig)})
#         # patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "seg": mostCommonClass(seg_pos), "label": label})
#         # patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "seg": mostCommonClass(seg_neg), "label": label})
#         # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
#         # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
#   return patches




def preproccessMINC():
  dir = '/home/yael/PycharmProjects/PatchDist/minc_cropped'
  path_to_merged = os.path.join(dir, 'merged')
  if os.path.isdir(path_to_merged):
    shutil.rmtree(path_to_merged)
  os.mkdir(path_to_merged)
  imgs = sorted(os.listdir(dir))[:-1]
  categories = [int(im.split('_')[0]) for im in imgs]
  cat_info = np.unique(categories, return_counts=True, return_index=True)
  newimid = 0
  for im in imgs:
    if cat_info[2][cat_info[0]==int(im.split('_')[0])]==1:
      continue
    anchor = plt.imread(os.path.join(dir, im))
    others = [img for img in imgs if img.split('_')[0]!=im.split('_')[0]]
    neg = random.sample(others, k=int(cat_info[2][cat_info[0]==int(im.split('_')[0])]-1))
    pos = [img for img in imgs if (img.split('_')[0]==im.split('_')[0] and img!=im)]
    # ims_same = random.sample(os.listdir(os.path.join(dir,str(f))), 5)
    # all_others_ = [[os.path.join(pth, file) for file in os.listdir(os.path.join(dir, pth))] for pth in folders if
    #                pth != f]
    # all_others = [item for sublist in all_others_ for item in sublist]
    # ims_diff = random.sample(all_others, 5)
    # # create training folder
    # img_list = os.listdir(os.path.join(dir, str(f)))
    # for im in img_list:
    #   img = plt.imread(os.path.join(dir, f, im))
    # center = anchor.shape / 2
    rect = [100,300,100,300]
    for sample in pos + neg:
      samp = plt.imread(os.path.join(dir, sample))
      new_im = copy.deepcopy(samp)
      new_im[rect[0]:rect[1], rect[2]:rect[3]] = anchor[rect[0]:rect[1], rect[2]:rect[3]]
      if new_im.shape != (400,400,3):
        continue
      imname = '{}_{}_{}.jpg'.format(im.split('_')[0], sample.split('_')[0], newimid)
      plt.imsave(os.path.join(dir, 'merged', imname), new_im)
      newimid += 1
  #   # img2_diff = plt.imread(os.path.join(dir,im))
  # images = [im for im in os.listdir(dir) if os.path.isfile(os.path.join(dir,im))]
  # doues = comb(images, 2)
  # counter = 0
  # for d in doues:
  #   counter += 1
  # idx = np.arange(counter)
  # np.random.shuffle(idx)
  # doues = comb(images, 2)
  # for i, dou in enumerate(doues):
  #   label = dou[0].split('_')[0] +'_'+ dou[1].split('_')[0]
  #   im1 = plt.imread(os.path.join(dir,dou[0]))
  #   im2 = plt.imread(os.path.join(dir,dou[1]))
  #   img = np.concatenate([im1, im2], axis=1)
  #   imname = dir+'/combined/{}_{}.jpg'.format(idx[i], label)
  #   plt.imsave(imname, img)


def preproccessMIT():
  data_dir = 'MIT_cropped'
  cats = os.listdir(data_dir)
  filenames_ = [os.listdir(os.path.join(data_dir, lab)) for lab in cats]
  images = [os.path.join(data_dir, cats[j], filenames_[j][i]) for j in xrange(0, len(cats)) for i in
            xrange(0, len(filenames_[j]))]
  # categories = [int(im.split('_')[0]) for im in imgs]
  # cat_info = np.unique(categories, return_counts=True, return_index=True)
  # newimid = 0
  for im in tqdm(images):
    try:
      img = Image.open(im).convert('RGB')
    except:
      os.remove(im)
      continue
    w, h = img.size
    min_dim = np.min([w,h])
    img = img.crop((0, 0, min_dim, min_dim))
    shape = (300, 300)
    img = img.resize(shape)
    img.save(im)


def preprocess():
  dir = 'imagenette2-320/clustering'
  for f in os.listdir(dir):
    folder = os.path.join(dir, f)
    for im in os.listdir(folder):
      orig = plt.imread(os.path.join(folder, im))
      if orig.shape[1]>orig.shape[0]:
        s = int(orig.shape[1]/2)-160
        cropped = orig[:,s:s+320,:]
      else:
        s = int(orig.shape[0] / 2) - 160
        cropped = orig[s:s + 320, :, :]
      plt.imsave(os.path.join(folder, im), cropped)


def getClassName(classID, cats):
  for i in range(len(cats)):
    if cats[i]['id'] == classID:
      return cats[i]['name']
  return "None"


def bsdsMask(dir):
  for fname in sorted(os.listdir(dir))[:25]:
    imname=fname.split('.')[0]+'.jpg'
    GT = loadmat(os.path.join(dir,fname))['groundTruth']
    person1 = GT[0, 0][0, 0][0]
    person2 = GT[0, 1][0, 0][0]
    person3 = GT[0, 2][0, 0][0]
    person4 = GT[0, 3][0, 0][0]
    plt.imshow(person1)
    mask = np.zeros_like(person1)
    mask[person1==pick]=1
    plt.imsave(os.path.join(dir, imname))



def preprocess_COCO(imgNames, coco):
  dir = 'COCO/clustering'
  if os.path.isdir(dir):
    shutil.rmtree(dir)
  os.mkdir(dir)
  dir_mask = 'COCO/clustering/mask'
  os.mkdir(dir_mask)
  catIDs = coco.getCatIds()
  cats = coco.loadCats(catIDs)
  filterClasses = ['bus', 'dog', 'pizza', 'scissors', 'airplane']
  # filterClasses = ['dog', 'bear', 'airplane', 'train', 'banana']
  # filterClasses = ['dog', 'elephant', 'fire hydrant', 'train', 'airplane']
  catIds = coco.getCatIds(catNms=filterClasses)
  images = []
  labels = []
  masks = []
  minh = np.inf
  minw = np.inf
  counter = 0
  for im in imgNames:
    # shutil.copy('COCO/train2017/{}'.format(im), os.path.join(dir, im))
    # img = plt.imread(os.path.join(dir, im))
    # print(img.shape)
    imgId = int(im.split('.')[0])
    try:
      img = coco.loadImgs(imgId)[0]
      # counter += 1
    except:
      continue
    if img['height']< minh:
      minh = img['height']
    if img['width'] < minw:
      minw = img['width']
  # shape = min(minh,minw)
  shape = 320
  for im in imgNames:
    try:
      # im = im.zfill(16)
      orig = plt.imread('COCO/train2017/{}'.format(im))
      counter += 1
      # try:
      #   orig = plt.imread('COCO/val2017/{}'.format(im))
      # except:
      #   continue
        # try:
      #     orig = plt.imread('COCO/test2017/{}'.format(im))
      im_shape_min = np.argmin(orig.shape[:2])
      if im_shape_min==0:
        new_shape = (int(orig.shape[1]/(orig.shape[0]/shape)), shape)
      else:
        new_shape = (shape, int(orig.shape[0] /(orig.shape[1]/shape)))
      if min(new_shape)<320:
        print(counter)# orig.resize(new_shape)
      orig = cv2.resize(orig, dsize=new_shape, interpolation=cv2.INTER_CUBIC)
      imrsz = orig[:shape, :shape, :]
      plt.imsave(os.path.join(dir, im), imrsz, dpi=1200)
      # img = plt.imread(os.path.join(dir, im))
      # print(img.shape)
      imgId = int(im.split('.')[0])
      img = coco.loadImgs(imgId)[0]
      mask = np.zeros((img['height'], img['width']))
      annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
      anns = coco.loadAnns(annIds)
      for i in range(len(anns)):
        className = getClassName(anns[i]['category_id'], cats)
        pixel_value = filterClasses.index(className) + 1
        mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)
      # mask = Image.fromarray(mask).resize((shape, shape))
      # mask = cv2.resize(mask, dsize=(shape, shape), interpolation=cv2.INTER_NEAREST)
        mask = Image.fromarray(mask)
        # mask.resize(new_shape)
        mask = np.array(mask)
      mask = cv2.resize(mask, dsize=new_shape, interpolation=cv2.INTER_NEAREST)
      plt.imsave(os.path.join(dir_mask,im), mask[:shape,:shape], cmap='gray')
      images.append(os.path.join(dir, im))
      masks.append(os.path.join(dir_mask, im))
      labels.append(className)
    except:
      continue
  return images, masks, labels
    # plt.imshow(mask)
    # folder = os.path.join(dir, f)
    # for im in os.listdir(folder):
    #
    #   if orig.shape[1]>orig.shape[0]:
    #     s = int(orig.shape[1]/2)-160
    #     cropped = orig[:,s:s+320,:]
    #   else:
    #     s = int(orig.shape[0] / 2) - 160
    #     cropped = orig[s:s + 320, :, :]
    #   plt.imsave(os.path.join(folder, im), cropped)


def preprocess_pascal():
  dir_im = 'VOC2012/JPEGImages'
  dir_mask = 'VOC2012/SegmentationClass'
  dir_im_proc = 'VOC2012/JPEGImages/preproc'
  dir_mask_proc ='VOC2012/SegmentationClass/preproc'
  # if os.path.isdir(dir):
  #   shutil.rmtree(dir)
  # os.mkdir(dir)
  # dir_mask = 'COCO/clustering/mask'
  # os.mkdir(dir_mask)
  # catIDs = coco.getCatIds()
  # cats = coco.loadCats(catIDs)
  # filterClasses = ['bus', 'dog', 'pizza', 'scissors', 'airplane']
  # # filterClasses = ['dog', 'bear', 'airplane', 'train', 'banana']
  # # filterClasses = ['dog', 'elephant', 'fire hydrant', 'train', 'airplane']
  # catIds = coco.getCatIds(catNms=filterClasses)
  images = []
  # labels = []
  masks = []
  imnames = []
  # minh = np.inf
  # minw = np.inf
  # counter = 0
  # for im in imgNames:
  #   # shutil.copy('COCO/train2017/{}'.format(im), os.path.join(dir, im))
  #   # img = plt.imread(os.path.join(dir, im))
  #   # print(img.shape)
  #   imgId = int(im.split('.')[0])
  #   try:
  #     # img = coco.loadImgs(imgId)[0]
  #     # counter += 1
  #   except:
  #     continue
  #   # if img['height']< minh:
  #   #   minh = img['height']
  #   # if img['width'] < minw:
  #     minw = img['width']
  # # shape = min(minh,minw)
  shape = 320
  counter = {}
  for im in os.listdir(dir_im):
    if not os.path.isfile(os.path.join(dir_im,im)) or im.split('.')[0]+'.png' not in os.listdir(dir_mask):
      continue
    try:
      # im = im.zfill(16)
      orig = plt.imread('VOC2012/JPEGImages/{}'.format(im))
      # img = plt.imread('VOC2012/SegmentationClass/{}'.format(im.split('.')[0]+'.png'))
      img = Image.open('VOC2012/SegmentationClass/{}'.format(im.split('.')[0]+'.png'))
      mask = np.array(img.getdata()).reshape((img.size[1], img.size[0]))
      # counter += 1
      # try:
      #   orig = plt.imread('VOC2012/JPEGImages/{}'.format(im))
      # except:
      #   continue
        # try:
      #     orig = plt.imread('COCO/test2017/{}'.format(im))
      im_shape_min = np.argmin(orig.shape[:2])
      if im_shape_min==0:
        new_shape = (int(orig.shape[1]/(orig.shape[0]/shape)), shape)
      else:
        new_shape = (shape, int(orig.shape[0] /(orig.shape[1]/shape)))
      # if min(new_shape)<320:
      #   print(counter)# orig.resize(new_shape)
      orig = cv2.resize(orig, dsize=new_shape, interpolation=cv2.INTER_CUBIC)
      mask = cv2.resize(mask, dsize=new_shape, interpolation=cv2.INTER_NEAREST)
      # mask = cv2.cvtColor(cv2.resize(mask, dsize=new_shape, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGBA2GRAY)
      # mn = mask.min()
      # mx = mask.max()
      # mx -= mn
      # mask = (((mask - mn) / mx) * 255).astype(np.uint8)

      imrsz = orig[:shape, :shape, :]
      maskrsz = mask[:shape, :shape]
      images.append(os.path.join(dir_im_proc, im))
      masks.append(os.path.join(dir_mask_proc, im.split('.')[0]+'.png'))
      plt.imsave(images[-1], imrsz, dpi=1200)
      plt.imsave(masks[-1], maskrsz)
      imnames.append(im)
    except:
      continue
  return images, masks, imnames
    # plt.imshow(mask)
    # folder = os.path.join(dir, f)
    # for im in os.listdir(folder):
    #
    #   if orig.shape[1]>orig.shape[0]:
    #     s = int(orig.shape[1]/2)-160
    #     cropped = orig[:,s:s+320,:]
    #   else:
    #     s = int(orig.shape[0] / 2) - 160
    #     cropped = orig[s:s + 320, :, :]
    #   plt.imsave(os.path.join(folder, im), cropped)


def preprocess_BSDS():
  dir = 'BSDS500/data/groundTruth/test'
  for file in os.listdir(dir):
    if '.mat' in file:
      mat = loadmat(os.path.join(dir,file))
      imname = file.split('.')[0]
      savename = os.path.join(dir,imname+'.npy')
      np.save(savename, mat)

  return


def getRep_imagenette():
  # preprocess()
  dir = 'imagenette2-320/clustering'
  folders = os.listdir(dir)
  images = []
  labels = []
  for l in folders:
    path = os.path.join(dir, l)
    filenames_ = os.listdir(path)
    filenames_.sort()
    img_lst = [os.path.join(dir, l, filenames_[i])
              for i in xrange(0, len(filenames_))]
    images.extend(img_lst)
    labels.extend([l for i in img_lst])

  queue = tf.compat.v1.train.slice_input_producer([images, labels], num_epochs=600, shuffle=False)
  image = tf.io.read_file(queue[0])
  label = queue[1]
  image = tf.image.decode_jpeg(image, channels=3)
  # shape = tf.shape(image)
  image.set_shape([320, 320, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)  # Changes to range [0,1)
  image = tf.expand_dims(image, 0)

  return image, label


def getCentroid(patch, patches):
  im = patch['im_id']
  patchesInIm = [p for p in patches if p['im_id']==im]
  for i, pa in enumerate(patchesInIm):
    pa['temp_idx'] = i
  dissim = np.zeros((len(patchesInIm), len(patchesInIm)))
  doues = comb(patchesInIm, 2)
  with multiprocessing.Pool(processes=16) as p:
    with tqdm(total=(len(patchesInIm) ** 2)) as pbar:
      for d in p.map(calcPatchDistmulti4centroid, doues):
        dissim[d[0], d[1]] = d[2]
        dissim[d[1], d[0]] = d[2]
  try:
    centroids = np.argpartition(np.mean(dissim, axis=1), 4)[:4]
  except:
    centroids = range(len(patchesInIm))
  return [p for p in patchesInIm if p['temp_idx'] in centroids], [p['idx'] for p in patchesInIm]




def getRep_coco():
  dataDir = './COCO'
  annFile = '{}/annotations/instances_train2017.json'.format(dataDir)
  # annFile = 'cocoanns.json'
  coco =COCO(annFile)

  with open('COCO/imgNames_newCOCO.pkl', 'rb') as f:
    imgNames = pickle.load(f)
  # with open('COCO/imgNames_largeCOCO.pkl', 'rb') as f:
  #   imgNames = pickle.load(f)
  # # imgNames = [im for im in os.listdir('COCO/clustering') if os.path.isfile(im)]
  # imgNames = ['22599.jpg', '206577.jpg', '386864.jpg', '161740.jpg', '307531.jpg', '339852.jpg', '375568.jpg', '546762.jpg',
  #             '522195.jpg', '269054.jpg', '355905.jpg', '176474.jpg', '107716.jpg', '481773.jpg','392186.jpg', '209289.jpg',
  #             '84187.jpg', '323602.jpg', '321057.jpg', '567997.jpg', '459712.jpg', '305690.jpg', '415360.jpg', '82680.jpg',
  #             '339470.jpg', '34824.jpg', '355240.jpg', '378403.jpg', '530154.jpg', '532381.jpg', '362567.jpg', '561887.jpg',
  #             '318908.jpg', '504341.jpg', '401059.jpg', '507198.jpg', '385921.jpg', '95044.jpg', '483069.jpg', '337525.jpg',
  #             '322500.jpg', '308143.jpg', '467469.jpg', '297394.jpg', '485897.jpg', '28874.jpg', '146623.jpg', '495048.jpg',
  #             '98709.jpg', '52891.jpg', '103584.jpg', '344460.jpg', '461505.jpg', '195297.jpg', '46345.jpg', '152488.jpg',
  #             '253981.jpg', '93816.jpg', '322945.jpg', '447200.jpg', '503235.jpg', '319142.jpg', '441674.jpg', '111548.jpg',
  #             '450747.jpg', '532575.jpg', '543210.jpg', '336741.jpg', '107375.jpg', '170326.jpg', '280858.jpg', '541944.jpg',
  #             '115652.jpg', '530059.jpg', '503050.jpg', '27617.jpg', '207289.jpg', '506783.jpg', '573277.jpg', '275885.jpg',
  #             '429638.jpg', '392818.jpg', '59741.jpg', '227687.jpg', '491660.jpg', '261824.jpg', '153960.jpg', '30494.jpg',
  #             '253785.jpg', '537041.jpg', '135032.jpg', '172017.jpg', '265003.jpg', '515226.jpg', '219632.jpg', '325380.jpg',#dogs
  #             '43873.jpg', '46245.jpg', '141468.jpg', '224878.jpg', '553162.jpg', '193435.jpg', '287095.jpg', '541880.jpg',
  #             '317241.jpg', '81476.jpg', '288592.jpg', '444087.jpg', '214280.jpg', '489183.jpg', '430805.jpg', '432675.jpg',
  #             '182927.jpg', '144915.jpg', '162767.jpg', '304214.jpg', '299357.jpg', '183359.jpg', '267624.jpg', '300123.jpg',
  #             '20972.jpg', '413871.jpg', '574942.jpg', '11552.jpg', '150812.jpg', '228004.jpg', '505655.jpg', '304220.jpg',
  #             '332562.jpg', '542720.jpg', '308476.jpg', '461507.jpg', '580698.jpg', '22002.jpg', '49184.jpg', '296511.jpg',
  #             '529379.jpg', '561027.jpg', '165141.jpg', '399752.jpg', '508015.jpg', '521667.jpg', '209326.jpg', '465347.jpg',
  #             '547519.jpg', '346295.jpg', '109542.jpg', '359337.jpg', '161937.jpg', '493067.jpg', '213437.jpg', '223950.jpg',
  #             '497350.jpg', '239706.jpg', '253831.jpg', '233948.jpg', '571695.jpg', '75948.jpg', '330368.jpg', '75090.jpg',
  #             '134846.jpg', '202966.jpg', '405527.jpg', '519611.jpg', '274184.jpg', '411436.jpg', '23433.jpg', '486240.jpg',
  #             '106243.jpg', '462984.jpg', '574720.jpg', '90244.jpg', '285908.jpg', '354326.jpg', '314376.jpg', '191686.jpg',
  #             '249623.jpg', '446764.jpg', '322392.jpg', '474502.jpg', '348585.jpg', '361180.jpg', '85163.jpg', '371183.jpg',
  #             '416145.jpg', '377294.jpg', '307583.jpg', '412939.jpg', '152898.jpg', '82091.jpg', '285089.jpg', '1442.jpg',#bears
  #             '237843.jpg', '567173.jpg', '255728.jpg', '486030.jpg', '322802.jpg', '62741.jpg', '456865.jpg', '204938.jpg',
  #             '183022.jpg', '200777.jpg', '497622.jpg', '33659.jpg', '371453.jpg', '254710.jpg', '123788.jpg', '249611.jpg',
  #             '308316.jpg', '235430.jpg', '249441.jpg', '488018.jpg', '235276.jpg', '404128.jpg', '442688.jpg', '489297.jpg',
  #             '8547.jpg', '162257.jpg', '231572.jpg', '21496.jpg', '149052.jpg', '255683.jpg', '8548.jpg', '297168.jpg',
  #             '245139.jpg', '359162.jpg', '517523.jpg', '527477.jpg', '167856.jpg', '470797.jpg', '436646.jpg', '413869.jpg',
  #             '492541.jpg', '575205.jpg', '374018.jpg', '263758.jpg', '457537.jpg', '247209.jpg', '52017.jpg', '27390.jpg',
  #             '532521.jpg', '411844.jpg', '203969.jpg', '246146.jpg', '16765.jpg', '135806.jpg', '310006.jpg', '183646.jpg',
  #             '82611.jpg', '358118.jpg', '308610.jpg', '482765.jpg', '390534.jpg', '277403.jpg', '342592.jpg', '543528.jpg',
  #             '549708.jpg', '253419.jpg', '279605.jpg', '524011.jpg', '574785.jpg', '115274.jpg', '262800.jpg', '281180.jpg',
  #             '116255.jpg', '435284.jpg', '500049.jpg', '490275.jpg', '348504.jpg', '336309.jpg', '547938.jpg', '414161.jpg',
  #             '581499.jpg', '527865.jpg', '64906.jpg', '498813.jpg', '162372.jpg', '551214.jpg', '191803.jpg', '364867.jpg',
  #             '119232.jpg', '377672.jpg', '524273.jpg', '467758.jpg', '255633.jpg', '193469.jpg', '85173.jpg', '488787.jpg', #airplanes
  #             '17198.jpg', '108819.jpg', '23995.jpg', '89861.jpg', '105261.jpg', '201386.jpg', '351283.jpg', '290952.jpg',
  #             '31666.jpg', '538105.jpg', '429586.jpg', '299448.jpg', '352498.jpg', '417217.jpg', '135618.jpg', '322049.jpg',
  #             '361417.jpg', '60990.jpg', '400104.jpg', '443533.jpg', '463605.jpg', '218862.jpg', '220368.jpg', '444363.jpg',
  #             '66325.jpg', '374533.jpg', '65357.jpg', '40011.jpg', '64859.jpg', '408863.jpg', '465618.jpg', '469135.jpg',
  #             '309585.jpg', '481732.jpg', '14622.jpg', '384023.jpg', '395957.jpg', '492102.jpg', '529193.jpg', '124294.jpg',
  #             '417201.jpg', '474271.jpg', '319266.jpg', '482298.jpg', '434022.jpg', '41843.jpg', '134717.jpg', '181386.jpg',
  #             '52974.jpg', '550466.jpg', '17328.jpg', '49562.jpg', '371945.jpg', '481759.jpg', '428168.jpg', '417726.jpg',
  #             '27504.jpg', '80096.jpg', '537861.jpg', '537589.jpg', '299862.jpg', '514519.jpg', '306373.jpg', '269680.jpg',
  #             '266586.jpg', '274606.jpg', '308110.jpg', '94168.jpg', '1374.jpg', '534259.jpg', '278032.jpg', '353357.jpg',
  #             '526806.jpg', '556372.jpg', '578967.jpg', '151264.jpg', '574856.jpg', '166541.jpg', '297510.jpg', '200612.jpg',
  #             '377878.jpg', '172173.jpg', '570463.jpg', '576505.jpg', '192001.jpg', '364743.jpg', '271190.jpg', '426061.jpg',
  #             '184321.jpg', '387672.jpg', '217197.jpg', '55166.jpg', '448710.jpg', '471280.jpg', '384395.jpg', '304552.jpg',#trains
  #             '113654.jpg', '499913.jpg', '3093.jpg', '157192.jpg', '457262.jpg', '295776.jpg', '99642.jpg', '125499.jpg',
  #             '164918.jpg', '316725.jpg', '191078.jpg', '360877.jpg', '511736.jpg', '347568.jpg', '41438.jpg', '266622.jpg',
  #             '546963.jpg', '19836.jpg', '54513.jpg', '114549.jpg', '484651.jpg', '72821.jpg', '326849.jpg', '156066.jpg',
  #             '47807.jpg', '140067.jpg', '139260.jpg', '271471.jpg', '95737.jpg', '112577.jpg', '337247.jpg', '533011.jpg',
  #             '556956.jpg', '144495.jpg', '78827.jpg', '436306.jpg', '238299.jpg', '125106.jpg', '528299.jpg', '527453.jpg',
  #             '357925.jpg', '460043.jpg', '49877.jpg', '516244.jpg', '138507.jpg', '4749.jpg', '425283.jpg', '396608.jpg',
  #             '486139.jpg', '325586.jpg', '291236.jpg', '427666.jpg', '7251.jpg', '370388.jpg', '259475.jpg', '429281.jpg',
  #             '472228.jpg', '59708.jpg', '430396.jpg', '51429.jpg', '308849.jpg', '59598.jpg', '267956.jpg', '397075.jpg',
  #             '148348.jpg', '67987.jpg', '207142.jpg', '161820.jpg', '500228.jpg', '263178.jpg', '117512.jpg', '12667.jpg',
  #             '215910.jpg', '473372.jpg', '124157.jpg', '3518.jpg', '9768.jpg', '450914.jpg', '563340.jpg', '372038.jpg',
  #             '433472.jpg', '35368.jpg', '491029.jpg', '504554.jpg', '459887.jpg', '420750.jpg', '58307.jpg', '543696.jpg',
  #             '114871.jpg', '43338.jpg', '96084.jpg', '265713.jpg', '197962.jpg', '112495.jpg', '157860.jpg', '320972.jpg']#bananas
  # with open('imgNames_largeCOCO.pkl', 'wb') as f:
  #   pickle.dump(imgNames, f)
  images, masks, labels = preprocess_COCO(imgNames, coco)

  # dir = 'imagenette2-320/clustering'
  # folders = os.listdir(dir)
  # images = []
  # labels = []
  # for l in folders:
  #   path = os.path.join(dir, l)
  #   filenames_ = os.listdir(path)
  #   filenames_.sort()
  #   img_lst = [os.path.join(dir, l, filenames_[i])
  #             for i in xrange(0, len(filenames_))]
  #   images.extend(img_lst)
  #   labels.extend([l for i in img_lst])

  queue = tf.compat.v1.train.slice_input_producer([images, masks, labels], num_epochs=600, shuffle=False)
  image = tf.io.read_file(queue[0])
  mask = tf.io.read_file(queue[1])
  label = queue[2]
  mask = tf.image.decode_jpeg(mask, channels=1)
  image = tf.image.decode_jpeg(image, channels=3)
  # shape = tf.shape(image)
  image.set_shape([320, 320, 3])
  mask.set_shape([320, 320, 1])
  image = tf.image.convert_image_dtype(image, tf.float32)
  ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  image = image - ds_mean
  image = tf.expand_dims(image, 0)

  return image, mask, label

def calcPatchDistmulti_(dou):
  # for dou in tqdm(doues):
  d, pr = dou
  p1, p2 = d
  patch1 = p1['patch']
  patch2 = p2['patch']
  if len(patch1.shape) > 2:
    patch1 = patch1.reshape(patch1.shape[0] * patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist = np.sqrt(np.around(distances(patch1, patch2), 3))
  dist1_vec = np.amin(dist, axis=0)
  dist2_vec = np.amin(dist, axis=1)
  # w1 = 1/(1+ np.argsort(dist1_vec))
  # w2 = 1/(1+ np.argsort(dist2_vec))
  # dist1 = np.average(dist1_vec, weights=w1)
  # dist2 = np.average(dist2_vec, weights=w2)
  dist1 = np.percentile(dist1_vec, pr)
  dist2 = np.percentile(dist2_vec, pr)
  # dist = distances(patch1, patch2)
  # dist1 = np.mean(np.sqrt(np.amin(dist, axis=0)))
  # dist2 = np.mean(np.sqrt(np.amin(dist, axis=1)))
  # dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  # dissim[p1['idx'], p2['idx']] = np.mean([dist1, dist2])
  # dissim[p2['idx'], p1['idx']] = np.mean([dist1, dist2])
  return p1['idx'], p2['idx'], np.mean(np.array([dist1, dist2])), np.array([dist1_vec, dist2_vec])


def calcPatchDistmulti_GPU(p1, p2, pr):
  # for dou in tqdm(doues):
  # (p1, p2) = dou
  patch1 = cp.asarray(p1['patch'])
  patch2 = cp.asarray(p2['patch'])

  if patch1.ndim > 2:
    patch1 = patch1.reshape(-1, patch1.shape[2])
    patch2 = patch2.reshape(-1, patch2.shape[2])
  dist = cp.sqrt(distances_GPU(patch1, patch2))
  # patch1 = cp.asarray(patch1)
  # patch2 = cp.asarray(patch2)

  # Compute pairwise L2 distances between all pixels
  # dists = cp.linalg.norm(patch1[:, None, :] - patch2[None, :, :], axis=2)  # shape [N1, N2]

  dist1_vec = cp.min(dist, axis=0)
  dist2_vec = cp.min(dist, axis=1)

  dist1 = cp.percentile(dist1_vec, pr)
  dist2 = cp.percentile(dist2_vec, pr)

  return p1['idx'], p2['idx'], float(cp.mean(cp.array([dist1, dist2])))

def aprioriPurity(patches, dissim):
  epsilon = int(0.005 * len(patches))
  purity = []
  for row in [p['idx'] for p in patches]:
    neighbours = np.array(patches)[np.argpartition(dissim[row], epsilon)[:epsilon]]
    purity.append(np.sum([p['seg']==patches[row]['seg'] for p in neighbours]) / epsilon)
  return purity

def dist_methods_comparison(patches):
  # patches = patches[:int(len(patches) * 0.8)]
  # patches = [p for p in patches if p['im_id']==range(10)]
  #generalized hausdorff
  quantiles = [10]
  # images = np.arange(0,150)
  score_haus = {}
  dists_same = []
  dists_notsame = []
  for q in quantiles:
    doues = comb(patches, 2)
    doues = [(d, q) for d in doues]
    dissim = np.zeros((len(patches), len(patches)))
    # zeroing = np.amin([p['idx'] for p in patches])
    with multiprocessing.Pool(processes=16) as p:
      with tqdm(total=(len(patches) ** 2) / 2) as pbar:
        for d in p.imap_unordered(calcPatchDistmulti_, doues):
          pbar.update()
          dissim[d[0], d[1]] = d[2]
          dissim[d[1], d[0]] = d[2]

    # b, w = dists_tot(patches, dissim)
    # dists_same.append(w)
    # dists_notsame.append(b)
    # score_haus[str(q)] = score(w, b)
    p.close()
    score_haus[str(q)] = aprioriPurity(patches, dissim)
    # score_haus[str(q)] = np.mean(aprioriPurity(patches, dissim))
  # lda
  solvers = ['svd', 'lsqr', 'eigen']
  scores_lda = {}
  for solver in solvers:
    # new_patches = random.sample(patches, int(len(patches) * 0.32))
    new_patches = patches
    lda = LDA(new_patches, solver)
    doues = comb(new_patches, 2)
    doues = [(d, lda) for d in doues]
    dists_same = []
    dists_notsame = []
    dissim = np.zeros((len(patches), len(patches)))
    with multiprocessing.Pool(processes=8) as p:
      with tqdm(total=(len(new_patches) ** 2)/2) as pbar:
        for d in p.imap_unordered(transorm2LDA, doues):
          pbar.update()
          if d[3]:
            if dissim[d[0], d[1]]:
              continue
            dissim[d[1], d[0]] = d[2]
            dissim[d[0], d[1]] = d[2]
            dists_same.append(d[2])
          else:
            if dissim[d[0], d[1]]:
              continue
            dissim[d[1], d[0]] = d[2]
            dissim[d[0], d[1]] = d[2]
            dists_notsame.append(d[2])
    # scores_lda[solver] = score_lda(dists_same, dists_notsame)
      purityList = aprioriPurity(patches, dissim)
      scores_lda[solver] = np.mean([item for item in purityList if item>0])
  #heuristic
  dists_same = []
  dists_notsame = []
  doues = comb(patches, 2)
  with multiprocessing.Pool(processes=8) as p:
    with tqdm(total=(len(patches) ** 2) / 2) as pbar:
      for d in p.imap_unordered(calcPatchDistmulti_avg, doues):
        pbar.update()
        dissim[d[1], d[0]] = d[2]
        dissim[d[0], d[1]] = d[2]
        if d[3]:
          dists_same.append(d[2])
        else:
          dists_notsame.append(d[2])
  # score_heu = score_lda(dists_same, dists_notsame)
  score_heu = aprioriPurity(patches, dissim)
  return score_heu, score_haus, scores_lda

def show_plot_safely():
  plt.draw()
  plt.pause(0.01)
  plt.show(block=True)


def dists_im(patches, im_idx, pr):
  classesIm = np.unique([p['im'] for p in patches], return_inverse=True,
                        return_counts=True)
  classesperIm = [[p for p in patches if p['im'] == imname] for imname in classesIm[0]]
  classesData = np.unique([p['seg'] for p in classesperIm[im_idx]], return_inverse=True,
                          return_counts=True)
  if len(classesData[0])==1:
    return None
  doues = comb(classesperIm[im_idx], 2)
  doues = [(d, pr) for d in doues]
  dissim = np.zeros((len(classesperIm[im_idx]), len(classesperIm[im_idx])))
  zeroing = np.amin([p['idx'] for p in classesperIm[im_idx]])
  with multiprocessing.Pool(processes=16) as p:
    # with tqdm(total=(len(classesperIm[im_idx]) ** 2) / 2) as pbar:
    for d in p.imap_unordered(calcPatchDistmulti_, doues):
      # pbar.update()
      dissim[d[0] - zeroing, d[1] - zeroing] = d[2]
      dissim[d[1] - zeroing, d[0] - zeroing] = d[2]

  patchesBySubclass = np.hstack([np.where(classesData[1] == i) for i in range(24)])
  dis4Hist = np.zeros(dissim.shape)
  for i in range(len(classesperIm[im_idx])):
    dis4Hist[i, :] = dissim[patchesBySubclass[0][i], :]
  dissim_forcopy = copy.deepcopy(dis4Hist)
  for i in range(len(classesperIm[im_idx])):
    dis4Hist[:, i] = dissim_forcopy[:, patchesBySubclass[0][i]]

  dists_within_class = []
  for i in range(len(classesData[2])):
    if i == 0:
      # continue
      dists_within_class.append(dis4Hist[:classesData[2][i], :classesData[2][i]])
    # elif i == 18:
    #   dists_within_class.append(dis4Hist[-classesData[2][i]:, -classesData[2][i]:])
    else:
      dists_within_class.append(dis4Hist[sum(classesData[2][:i]):sum(classesData[2][:(i + 1)]),
                                sum(classesData[2][:i]):sum(classesData[2][:(i + 1)])])

  dists_between_class = []
  for i in range(len(classesData[2])):
    if i == 0:
      dists_between_class.append(dis4Hist[:classesData[2][i], classesData[2][i]:])
    # elif i == 18:
    #   dists_between_class.append(dis4Hist[-classesData[2][i]:, :-classesData[2][i]])
    else:
      dists_between_class.append(np.hstack((dis4Hist[sum(classesData[2][:i]):sum(classesData[2][:(i + 1)]),
                                            :sum(classesData[2][:i])],
                                            dis4Hist[sum(classesData[2][:i]):sum(classesData[2][:(i + 1)]),
                                            sum(classesData[2][:(i + 1)]):])))
  return dists_between_class, dists_within_class, classesperIm


def dists_tot(patches, dissim):
  # classesIm = np.unique([p['im'] for p in patches], return_inverse=True,
  #                       return_counts=True)
  # classesperIm = [[p for p in patches if p['im'] == imname] for imname in classesIm[0]]
  classesData = np.unique([p['seg'] for p in patches], return_inverse=True, return_counts=True)
  # doues = comb(patches, 2)
  # doues = [(d, pr) for d in doues]
  # dissim = np.zeros((len(patches), len(patches)))
  # # zeroing = np.amin([p['idx'] for p in patches])
  # with multiprocessing.Pool(processes=16) as p:
  #   # with tqdm(total=(len(classesperIm[im_idx]) ** 2) / 2) as pbar:
  #   for d in p.imap_unordered(calcPatchDistmulti_, doues):
  #     # pbar.update()
  #     dissim[d[0], d[1]] = d[2]
  #     dissim[d[1], d[0]] = d[2]

  patchesBySubclass = np.hstack([np.where(classesData[0][classesData[1]] == seg) for seg in classesData[0]])
  dis4Hist = np.zeros(dissim.shape)
  for i in range(len(classesData[1])):
    dis4Hist[i, :] = dissim[patchesBySubclass[0][i], :]
  dissim_forcopy = copy.deepcopy(dis4Hist)
  for i in range(len(classesData[1])):
    dis4Hist[:, i] = dissim_forcopy[:, patchesBySubclass[0][i]]

  dists_within_class = []
  for i in range(len(classesData[2])):
    if i == 0:
      # continue
      dists_within_class.append(dis4Hist[:classesData[2][i], :classesData[2][i]])
    # elif i == 18:
    #   dists_within_class.append(dis4Hist[-classesData[2][i]:, -classesData[2][i]:])
    else:
      dists_within_class.append(dis4Hist[sum(classesData[2][:i]):sum(classesData[2][:(i + 1)]),
                                sum(classesData[2][:i]):sum(classesData[2][:(i + 1)])])

  dists_between_class = []
  for i in range(len(classesData[2])):
    if i == 0:
      dists_between_class.append(dis4Hist[:classesData[2][i], classesData[2][i]:])
    # elif i == 18:
    #   dists_between_class.append(dis4Hist[-classesData[2][i]:, :-classesData[2][i]])
    else:
      dists_between_class.append(np.hstack((dis4Hist[sum(classesData[2][:i]):sum(classesData[2][:(i + 1)]),
                                            :sum(classesData[2][:i])],
                                            dis4Hist[sum(classesData[2][:i]):sum(classesData[2][:(i + 1)]),
                                            sum(classesData[2][:(i + 1)]):])))
  return dists_between_class, dists_within_class

def score(dists_within_class, dists_between_class):
  tot_w_mean = np.mean(np.hstack([dists_within_class[i][np.triu_indices(dists_within_class[i].shape[0])] for i in range(len(dists_within_class))]))
  tot_w_std = np.std(np.hstack([dists_within_class[i][np.triu_indices(dists_within_class[i].shape[0])] for i in range(len(dists_within_class))]))
  tot_b_mean = np.mean(np.hstack([dists_between_class[i].flatten() for i in range(len(dists_between_class))]))
  tot_b_std = np.std(np.hstack([dists_between_class[i].flatten() for i in range(len(dists_between_class))]))
  score = 0.25*(tot_w_mean - tot_b_mean) ** 2 / (tot_w_std**2 + tot_b_std**2) + 1 / 2 * np.log(
    (tot_w_std**2 + tot_b_std**2) / (2 * tot_w_std *tot_b_std))
  return score

def score_lda(dists_within_class, dists_between_class):
  tot_w_mean = np.mean(dists_within_class)
  tot_w_std = np.std(dists_within_class)
  tot_b_mean = np.mean(dists_between_class)
  tot_b_std = np.std(dists_between_class)
  score = 0.25*(tot_w_mean - tot_b_mean) ** 2 / (tot_w_std**2 + tot_b_std**2) + 1 / 2 * np.log(
    (tot_w_std**2 + tot_b_std**2) / (2 * tot_w_std *tot_b_std))
  return score

def rgb_to_label(segm):
  _COLORS = tf.constant([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
  ], dtype=tf.int32)
  segm = tf.cast(segm[..., tf.newaxis], _COLORS.dtype)
  return tf.argmax(tf.reduce_all(segm == tf.transpose(_COLORS), axis=-2), axis=-1)
  # return segm


def getRep_pascal():
  # dataDir = './VOC2012'
  # annFile = '{}/annotations/instances_train2017.json'.format(dataDir)
  # # annFile = 'cocoanns.json'
  # coco =COCO(annFile)
  #
  # with open('COCO/imgNames_newCOCO.pkl', 'rb') as f:
  #   imgNames = pickle.load(f)
  # # with open('COCO/imgNames_largeCOCO.pkl', 'rb') as f:
  # #   imgNames = pickle.load(f)
  # # # imgNames = [im for im in os.listdir('COCO/clustering') if os.path.isfile(im)]
  # # imgNames = ['22599.jpg', '206577.jpg', '386864.jpg', '161740.jpg', '307531.jpg', '339852.jpg', '375568.jpg', '546762.jpg',
  # #             '522195.jpg', '269054.jpg', '355905.jpg', '176474.jpg', '107716.jpg', '481773.jpg','392186.jpg', '209289.jpg',
  # #             '84187.jpg', '323602.jpg', '321057.jpg', '567997.jpg', '459712.jpg', '305690.jpg', '415360.jpg', '82680.jpg',
  # #             '339470.jpg', '34824.jpg', '355240.jpg', '378403.jpg', '530154.jpg', '532381.jpg', '362567.jpg', '561887.jpg',
  # #             '318908.jpg', '504341.jpg', '401059.jpg', '507198.jpg', '385921.jpg', '95044.jpg', '483069.jpg', '337525.jpg',
  # #             '322500.jpg', '308143.jpg', '467469.jpg', '297394.jpg', '485897.jpg', '28874.jpg', '146623.jpg', '495048.jpg',
  # #             '98709.jpg', '52891.jpg', '103584.jpg', '344460.jpg', '461505.jpg', '195297.jpg', '46345.jpg', '152488.jpg',
  # #             '253981.jpg', '93816.jpg', '322945.jpg', '447200.jpg', '503235.jpg', '319142.jpg', '441674.jpg', '111548.jpg',
  # #             '450747.jpg', '532575.jpg', '543210.jpg', '336741.jpg', '107375.jpg', '170326.jpg', '280858.jpg', '541944.jpg',
  # #             '115652.jpg', '530059.jpg', '503050.jpg', '27617.jpg', '207289.jpg', '506783.jpg', '573277.jpg', '275885.jpg',
  # #             '429638.jpg', '392818.jpg', '59741.jpg', '227687.jpg', '491660.jpg', '261824.jpg', '153960.jpg', '30494.jpg',
  # #             '253785.jpg', '537041.jpg', '135032.jpg', '172017.jpg', '265003.jpg', '515226.jpg', '219632.jpg', '325380.jpg',#dogs
  # #             '43873.jpg', '46245.jpg', '141468.jpg', '224878.jpg', '553162.jpg', '193435.jpg', '287095.jpg', '541880.jpg',
  # #             '317241.jpg', '81476.jpg', '288592.jpg', '444087.jpg', '214280.jpg', '489183.jpg', '430805.jpg', '432675.jpg',
  # #             '182927.jpg', '144915.jpg', '162767.jpg', '304214.jpg', '299357.jpg', '183359.jpg', '267624.jpg', '300123.jpg',
  # #             '20972.jpg', '413871.jpg', '574942.jpg', '11552.jpg', '150812.jpg', '228004.jpg', '505655.jpg', '304220.jpg',
  # #             '332562.jpg', '542720.jpg', '308476.jpg', '461507.jpg', '580698.jpg', '22002.jpg', '49184.jpg', '296511.jpg',
  # #             '529379.jpg', '561027.jpg', '165141.jpg', '399752.jpg', '508015.jpg', '521667.jpg', '209326.jpg', '465347.jpg',
  # #             '547519.jpg', '346295.jpg', '109542.jpg', '359337.jpg', '161937.jpg', '493067.jpg', '213437.jpg', '223950.jpg',
  # #             '497350.jpg', '239706.jpg', '253831.jpg', '233948.jpg', '571695.jpg', '75948.jpg', '330368.jpg', '75090.jpg',
  # #             '134846.jpg', '202966.jpg', '405527.jpg', '519611.jpg', '274184.jpg', '411436.jpg', '23433.jpg', '486240.jpg',
  # #             '106243.jpg', '462984.jpg', '574720.jpg', '90244.jpg', '285908.jpg', '354326.jpg', '314376.jpg', '191686.jpg',
  # #             '249623.jpg', '446764.jpg', '322392.jpg', '474502.jpg', '348585.jpg', '361180.jpg', '85163.jpg', '371183.jpg',
  # #             '416145.jpg', '377294.jpg', '307583.jpg', '412939.jpg', '152898.jpg', '82091.jpg', '285089.jpg', '1442.jpg',#bears
  # #             '237843.jpg', '567173.jpg', '255728.jpg', '486030.jpg', '322802.jpg', '62741.jpg', '456865.jpg', '204938.jpg',
  # #             '183022.jpg', '200777.jpg', '497622.jpg', '33659.jpg', '371453.jpg', '254710.jpg', '123788.jpg', '249611.jpg',
  # #             '308316.jpg', '235430.jpg', '249441.jpg', '488018.jpg', '235276.jpg', '404128.jpg', '442688.jpg', '489297.jpg',
  # #             '8547.jpg', '162257.jpg', '231572.jpg', '21496.jpg', '149052.jpg', '255683.jpg', '8548.jpg', '297168.jpg',
  # #             '245139.jpg', '359162.jpg', '517523.jpg', '527477.jpg', '167856.jpg', '470797.jpg', '436646.jpg', '413869.jpg',
  # #             '492541.jpg', '575205.jpg', '374018.jpg', '263758.jpg', '457537.jpg', '247209.jpg', '52017.jpg', '27390.jpg',
  # #             '532521.jpg', '411844.jpg', '203969.jpg', '246146.jpg', '16765.jpg', '135806.jpg', '310006.jpg', '183646.jpg',
  # #             '82611.jpg', '358118.jpg', '308610.jpg', '482765.jpg', '390534.jpg', '277403.jpg', '342592.jpg', '543528.jpg',
  # #             '549708.jpg', '253419.jpg', '279605.jpg', '524011.jpg', '574785.jpg', '115274.jpg', '262800.jpg', '281180.jpg',
  # #             '116255.jpg', '435284.jpg', '500049.jpg', '490275.jpg', '348504.jpg', '336309.jpg', '547938.jpg', '414161.jpg',
  # #             '581499.jpg', '527865.jpg', '64906.jpg', '498813.jpg', '162372.jpg', '551214.jpg', '191803.jpg', '364867.jpg',
  # #             '119232.jpg', '377672.jpg', '524273.jpg', '467758.jpg', '255633.jpg', '193469.jpg', '85173.jpg', '488787.jpg', #airplanes
  # #             '17198.jpg', '108819.jpg', '23995.jpg', '89861.jpg', '105261.jpg', '201386.jpg', '351283.jpg', '290952.jpg',
  # #             '31666.jpg', '538105.jpg', '429586.jpg', '299448.jpg', '352498.jpg', '417217.jpg', '135618.jpg', '322049.jpg',
  # #             '361417.jpg', '60990.jpg', '400104.jpg', '443533.jpg', '463605.jpg', '218862.jpg', '220368.jpg', '444363.jpg',
  # #             '66325.jpg', '374533.jpg', '65357.jpg', '40011.jpg', '64859.jpg', '408863.jpg', '465618.jpg', '469135.jpg',
  # #             '309585.jpg', '481732.jpg', '14622.jpg', '384023.jpg', '395957.jpg', '492102.jpg', '529193.jpg', '124294.jpg',
  # #             '417201.jpg', '474271.jpg', '319266.jpg', '482298.jpg', '434022.jpg', '41843.jpg', '134717.jpg', '181386.jpg',
  # #             '52974.jpg', '550466.jpg', '17328.jpg', '49562.jpg', '371945.jpg', '481759.jpg', '428168.jpg', '417726.jpg',
  # #             '27504.jpg', '80096.jpg', '537861.jpg', '537589.jpg', '299862.jpg', '514519.jpg', '306373.jpg', '269680.jpg',
  # #             '266586.jpg', '274606.jpg', '308110.jpg', '94168.jpg', '1374.jpg', '534259.jpg', '278032.jpg', '353357.jpg',
  # #             '526806.jpg', '556372.jpg', '578967.jpg', '151264.jpg', '574856.jpg', '166541.jpg', '297510.jpg', '200612.jpg',
  # #             '377878.jpg', '172173.jpg', '570463.jpg', '576505.jpg', '192001.jpg', '364743.jpg', '271190.jpg', '426061.jpg',
  # #             '184321.jpg', '387672.jpg', '217197.jpg', '55166.jpg', '448710.jpg', '471280.jpg', '384395.jpg', '304552.jpg',#trains
  # #             '113654.jpg', '499913.jpg', '3093.jpg', '157192.jpg', '457262.jpg', '295776.jpg', '99642.jpg', '125499.jpg',
  # #             '164918.jpg', '316725.jpg', '191078.jpg', '360877.jpg', '511736.jpg', '347568.jpg', '41438.jpg', '266622.jpg',
  # #             '546963.jpg', '19836.jpg', '54513.jpg', '114549.jpg', '484651.jpg', '72821.jpg', '326849.jpg', '156066.jpg',
  # #             '47807.jpg', '140067.jpg', '139260.jpg', '271471.jpg', '95737.jpg', '112577.jpg', '337247.jpg', '533011.jpg',
  # #             '556956.jpg', '144495.jpg', '78827.jpg', '436306.jpg', '238299.jpg', '125106.jpg', '528299.jpg', '527453.jpg',
  # #             '357925.jpg', '460043.jpg', '49877.jpg', '516244.jpg', '138507.jpg', '4749.jpg', '425283.jpg', '396608.jpg',
  # #             '486139.jpg', '325586.jpg', '291236.jpg', '427666.jpg', '7251.jpg', '370388.jpg', '259475.jpg', '429281.jpg',
  # #             '472228.jpg', '59708.jpg', '430396.jpg', '51429.jpg', '308849.jpg', '59598.jpg', '267956.jpg', '397075.jpg',
  # #             '148348.jpg', '67987.jpg', '207142.jpg', '161820.jpg', '500228.jpg', '263178.jpg', '117512.jpg', '12667.jpg',
  # #             '215910.jpg', '473372.jpg', '124157.jpg', '3518.jpg', '9768.jpg', '450914.jpg', '563340.jpg', '372038.jpg',
  # #             '433472.jpg', '35368.jpg', '491029.jpg', '504554.jpg', '459887.jpg', '420750.jpg', '58307.jpg', '543696.jpg',
  # #             '114871.jpg', '43338.jpg', '96084.jpg', '265713.jpg', '197962.jpg', '112495.jpg', '157860.jpg', '320972.jpg']#bananas
  # # with open('imgNames_largeCOCO.pkl', 'wb') as f:
  # #   pickle.dump(imgNames, f)
  images, masks, imnames = preprocess_pascal()

  # dir = 'imagenette2-320/clustering'
  # folders = os.listdir(dir)
  # images = []
  # labels = []
  # for l in folders:
  #   path = os.path.join(dir, l)
  #   filenames_ = os.listdir(path)
  #   filenames_.sort()
  #   img_lst = [os.path.join(dir, l, filenames_[i])
  #             for i in xrange(0, len(filenames_))]
  #   images.extend(img_lst)
  #   labels.extend([l for i in img_lst])

  queue = tf.compat.v1.train.slice_input_producer([images, masks, imnames], num_epochs=600, shuffle=False)
  image = tf.io.read_file(queue[0])
  mask = tf.io.read_file(queue[1])
  imname = queue[2]
  # label = queue[2]
  mask = tf.image.decode_png(mask, channels=0)
  image = tf.image.decode_jpeg(image, channels=3)
  #PASCAL_VOC_classes = { 0: 'background', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'table', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'potted_plant', 17: 'sheep', 18: 'sofa', 19 : 'train', 20 : 'tv', 21 : 'void' }


  mask = tf.cast(mask, tf.uint8)
  # shape = tf.shape(image)
  image.set_shape([320, 320, 3])
  mask.set_shape([320, 320,1])
  # mask = rgb_to_label(mask)

  image = tf.image.convert_image_dtype(image, tf.float32)
  ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  # image = image - ds_mean
  image = tf.expand_dims(image, 0)

  return image, mask, imname


def getRep_MINC():
  # preproccessMINC()
  data_dir = 'minc_cropped/merged'
  # data_dir = 'rectangleExp/bsds'
  filenames_ = os.listdir(data_dir)
  # filenames_.sort(key=lambda x:x.split('_')[2])
  images = [os.path.join(data_dir, filenames_[i]) for i in xrange(0, len(filenames_))]
  labels = [imname.split('_')[:2] for imname in filenames_]
  # imNames = os.listdir(dataDir)
  # images = [os.path.join(dataDir, im) for im in imNames]
  queue = tf.compat.v1.train.slice_input_producer([images, labels], num_epochs=600, shuffle=False)
  image = tf.io.read_file(queue[0])
  # mask = tf.io.read_file(queue[1])
  label = queue[1]
  # mask = tf.image.decode_jpeg(mask, channels=1)
  image = tf.image.decode_jpeg(image, channels=3)
  # shape = tf.shape(image)
  # image.set_shape([320, 479, 3])
  image.set_shape([400, 400, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  bsds_mean = np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
  # iamgenette_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  # image = image - bsds_mean
  # ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  # image = image - ds_mean
  image = tf.expand_dims(image, 0)

  return image, label


def getRep_MIT():
  # preproccessMIT()
  data_dir = 'MIT_cropped'
  # data_dir = 'rectangleExp/bsds'
  pot_labels = os.listdir(data_dir)
  filenames_ = [os.listdir(os.path.join(data_dir, lab)) for lab in pot_labels]
  # filenames_.sort(key=lambda x:x.split('_')[2])
  images = [os.path.join(data_dir, pot_labels[j], filenames_[j][i]) for j in xrange(0, len(pot_labels)) for i in xrange(0, len(filenames_[j]))]
  labels = [imname.split('/')[1] for imname in images]
  # imNames = os.listdir(dataDir)
  # images = [os.path.join(dataDir, im) for im in imNames]
  # labels = [pot_labels[i]* len(filenames_[i]) for i in xrange(0, len(pot_labels))]
  queue = tf.compat.v1.train.slice_input_producer([images, labels], num_epochs=600, shuffle=False)
  image = tf.io.read_file(queue[0])
  # mask = tf.io.read_file(queue[1])
  label = queue[1]
  imname = queue[0]
  # mask = tf.image.decode_jpeg(mask, channels=1)
  image = tf.image.decode_jpeg(image, channels=3)
  # shape = tf.shape(image)
  # image.set_shape([320, 479, 3])
  image.set_shape([300, 300, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  bsds_mean = np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
  # iamgenette_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  # image = image - bsds_mean
  # ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  # image = image - ds_mean
  image = tf.expand_dims(image, 0)

  return image, label, imname


def getRep_Rect():
  ###
  data_dir = 'rectangleExp/bsds'
  # data_dir = 'rectangleExp/bsds'
  filenames_ = os.listdir(data_dir)
  filenames_.sort()
  images = [os.path.join(data_dir, filenames_[i]) for i in xrange(0, len(filenames_))]
  # imNames = os.listdir(dataDir)
  # images = [os.path.join(dataDir, im) for im in imNames]
  queue = tf.compat.v1.train.slice_input_producer([images], num_epochs=600, shuffle=False)
  image = tf.io.read_file(queue[0])
  # mask = tf.io.read_file(queue[1])
  # label = queue[2]
  # mask = tf.image.decode_jpeg(mask, channels=1)
  image = tf.image.decode_jpeg(image, channels=3)
  # shape = tf.shape(image)
  # image.set_shape([320, 479, 3])
  image.set_shape([321, 481, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  bsds_mean = np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
  # iamgenette_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  image = image - bsds_mean
  # ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  # image = image - ds_mean
  image = tf.expand_dims(image, 0)

  return image

def process_group(group):
  results = [calcPatchDistmulti(pair) for pair in group]
  dissim_in = sorted(results, key=lambda x: x[2])[:1000]
  return dissim_in
@jit(parallel=True)
def calcdissim(patches):
  dissim = np.zeros((len(patches), len(patches)))
  for i in range(len(patches)):
    for j in tqdm(range(len(patches))):
      p1 = patches[i]
      p2 = patches[j]
      if dissim[p1['idx'], p2['idx']] != 0:
        continue
      else:
        # start = time.time()
        # patch1 = torch.from_numpy(p1['patch'])
        # patch2 = torch.from_numpy(p2['patch'])
        # dist = calcPatchDisttorch(patch1, patch2)
        dist = calcPatchDist(p1['patch'], p2['patch'])
        dissim[p1['idx'], p2['idx']] = dist
        dissim[p2['idx'], p1['idx']] = dist
  return dissim


# class metric1(nn.Module()):
#   def __init__(self, p1, p2):
#     super(Model, self).__init__()
#     self.p1 = p1['patch']
#     self.p2 = p2['patch']
#
#   def calcPatchD(self, p, patches):
#
#   def nearest_redi(X, Y):
#     dist = np.zeros((X.shape[0], Y.shape[0]))
#     dist = (X ** 2).sum(axis=1)[:, np.newaxis] + (Y ** 2).sum(axis=1) - 2 * X.dot(Y.T)
#   def forward(self, input):
#     output = self.fc(input)
#     print("\tIn Model: input size", input.size(),
#           "output size", output.size())
#
#     return output
#

def distsY(dou):
  p1, p2 = dou
  return np.linalg.norm(p1[1] - p2[1]), p1[0], p2[0]

def batch_iterable(iterable, batch_size):
  it = iter(iterable)
  while True:
    batch = list(islice(it, batch_size))
    if not batch:
      break
    yield batch


def compute_dissim_gpu(patches, batch_size=50000, percentile=10):
  N = len(patches)
  dissim = np.zeros((N, N), dtype=np.float32)

  doues = comb(patches, 2)
  total_pairs = N * (N - 1) // 2

  for batch in tqdm(batch_iterable(doues, batch_size), total=total_pairs // batch_size + 1):
    for p1, p2 in batch:
      i, j, d = calcPatchDistmulti_GPU(p1, p2, pr=percentile)
      dissim[i, j] = d
      dissim[j, i] = d  # symmetric

  return dissim

def prepare_patch_tensor(patches):
  reshaped = []
  for p in patches:
    patch = p['patch']
    if patch.ndim > 2:
      patch = patch.reshape(-1, patch.shape[2])  # flatten spatial dims
    reshaped.append(patch)
  return reshaped  # list of [N_i, D] arrays

def compute_hausdorff_batch(patches, idx_batch, percentile=10):
  import cupy as cp  # Import inside the function to ensure local initialization

  results = []
  for i, j in idx_batch:
    # Get raw CPU patch data from the passed patches list
    p1 = patches[i]['patch']
    p2 = patches[j]['patch']

    if p1.ndim > 2:
      p1 = p1.reshape(-1, p1.shape[2])
      p2 = p2.reshape(-1, p2.shape[2])

    # Now, initialize GPU arrays here
    X = cp.asarray(p1)
    Y = cp.asarray(p2)

    # Compute distances as before
    dist = cp.sum(X ** 2, axis=1)[:, None] + cp.sum(Y ** 2, axis=1)[None, :] - 2 * (X @ Y.T)
    dist = cp.sqrt(cp.maximum(dist, 0))

    d1_vec = cp.min(dist, axis=0)
    d2_vec = cp.min(dist, axis=1)

    d1 = cp.percentile(d1_vec, percentile)
    d2 = cp.percentile(d2_vec, percentile)

    results.append((i, j, float(cp.mean(cp.array([d1, d2])))))
  return results

  def batch_iterable(iterable, batch_size):
    it = iter(iterable)
    while True:
      batch = list(islice(it, batch_size))
      if not batch:
        break
      yield batch


def compute_dissim_parallel(patches, batch_size=5000, percentile=10, max_workers=8):
  N = len(patches)
  dissim = np.zeros((N, N), dtype=np.float32)
  patch_tensor = prepare_patch_tensor(patches)

  pair_iter = comb(range(N), 2)
  total_pairs = N * (N - 1) // 2
  batches = list(batch_iterable(pair_iter, batch_size))

  ctx = get_context("spawn")
  with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as executor:
    futures = {
      executor.submit(compute_hausdorff_batch, patch_tensor, batch, percentile): batch
      for batch in batches
    }

    for f in tqdm(as_completed(futures), total=len(futures)):
      results = f.result()
      for i, j, d in results:
        dissim[i, j] = d
        dissim[j, i] = d

  return dissim

from concurrent.futures import ThreadPoolExecutor, as_completed
def compute_dissim_parallel_threads(patches, batch_size=5000, percentile=10, max_workers=8):
  N = len(patches)
  dissim = np.memmap('dissim.npy', dtype='float32', mode='w+', shape=(N, N))

  from itertools import combinations, islice

  def batch_iterable(iterable, batch_size):
    it = iter(iterable)
    while True:
      batch = list(islice(it, batch_size))
      if not batch:
        break
      yield batch

  # Create batches of index pairs
  pair_iter = combinations(range(N), 2)
  batches = list(batch_iterable(pair_iter, batch_size))

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(compute_hausdorff_batch, patches, batch, percentile)
               for batch in batches]

    for future in tqdm(as_completed(futures), total=len(futures)):
      results = future.result()
      for i, j, d in results:
        dissim[i, j] = d
        dissim[j, i] = d
      cp.get_default_memory_pool().free_all_blocks()

  return dissim

def calc_distance_gpu(p1, p2, percentile=10):
  """
  Compute the Hausdorff-like distance between patch p1 and p2 on the GPU.
  p1 and p2 are dictionaries with key 'patch' containing a NumPy array.
  """
  patch1 = p1['patch']
  patch2 = p2['patch']

  # If the patch has spatial dimensions, flatten it:
  if patch1.ndim > 2:
    patch1 = patch1.reshape(-1, patch1.shape[2])
    patch2 = patch2.reshape(-1, patch2.shape[2])

  # Transfer data to GPU
  X = cp.asarray(patch1)
  Y = cp.asarray(patch2)

  # Compute Euclidean distances using vectorized operations
  dist = cp.sum(X ** 2, axis=1)[:, None] + cp.sum(Y ** 2, axis=1)[None, :] - 2 * (X @ Y.T)
  dist = cp.sqrt(cp.maximum(dist, 0))

  # Compute vectors of minimal distances along each axis
  d1_vec = cp.min(dist, axis=0)
  d2_vec = cp.min(dist, axis=1)

  d1 = cp.percentile(d1_vec, percentile)
  d2 = cp.percentile(d2_vec, percentile)

  # Return the average of the two distances (as a CPU float)
  return float(cp.mean(cp.array([d1, d2])))

def compute_all_dissim(patches, block_size=1000, percentile=10, memmap_filename='dissim.npy'):
  """
  Compute the full dissimilarity matrix for a list of patches using block processing.

  Parameters:
    patches      : list of patch dictionaries.
    block_size   : number of patches to process at a time along one dimension.
    percentile   : percentile value used in the distance metric.
    memmap_filename : filename for the memory-mapped output.

  Returns:
    dissim       : a NumPy memmap array containing the full dissimilarity matrix.
  """
  N = len(patches)
  # Create a memmap file large enough to hold the full matrix.
  dissim = np.memmap(memmap_filename, dtype='float32', mode='w+', shape=(N, N))

  # Process blocks: for each block, compute pairwise distances.
  for i_start in range(0, N, block_size):
    i_end = min(N, i_start + block_size)
    for j_start in range(i_start, N, block_size):
      j_end = min(N, j_start + block_size)
      print(f"Processing block: rows {i_start} to {i_end}, cols {j_start} to {j_end}")

      # Allocate block for current submatrix
      block = np.zeros((i_end - i_start, j_end - j_start), dtype=np.float32)

      # Compute distances for each pair in this block
      for ii, i in enumerate(range(i_start, i_end)):
        # When j < i, the value has been computed already; reuse symmetry.
        for jj, j in enumerate(range(j_start, j_end)):
          if j < i:
            block[ii, jj] = dissim[j, i]
          elif i == j:
            block[ii, jj] = 0.
          else:
            # Compute the distance for this pair on GPU.
            d = calc_distance_gpu(patches[i], patches[j], percentile=percentile)
            block[ii, jj] = d

      # Write the computed block into the memmap
      dissim[i_start:i_end, j_start:j_end] = block
      # Fill the symmetric block if not on the diagonal
      if i_start != j_start:
        dissim[j_start:j_end, i_start:i_end] = block.T

      # Free GPU memory caches in case they are still held:
      cp.get_default_memory_pool().free_all_blocks()

  # Flush changes to disk before returning
  dissim.flush()
  return dissim


# ---------------------------
# Function to compute a block of the dissimilarity matrix.
# Each block corresponds to rows [i_start:i_end] and cols [j_start:j_end].
def compute_block(args):
  # Unpack arguments: patches is the raw CPU list; indices are the block ranges.
  patches, i_start, i_end, j_start, j_end, percentile = args
  block_rows = i_end - i_start
  block_cols = j_end - j_start
  # We'll store the block in a NumPy array (on CPU) because we'll return it.
  block = np.zeros((block_rows, block_cols), dtype=np.float32)

  # Compute pairwise distances for the given block.
  # (Assuming that for i==j we set 0 and for j<i we leave the block to be filled from symmetry.)
  for ii, i in enumerate(range(i_start, i_end)):
    for jj, j in enumerate(range(j_start, j_end)):
      if j < i:
        continue  # we will fill symmetric values later in main process.
      elif i == j:
        block[ii, jj] = 0.
      else:
        try:
          d = calc_distance_gpu(patches[i], patches[j], percentile=percentile)
        except Exception as e:
          print(f"Error computing distance for patches {i}, {j}: {e}")
          d = np.nan
        block[ii, jj] = d
  # Return block indices and the computed block.
  return i_start, i_end, j_start, j_end, block


# ---------------------------
# Function to compute the full dissimilarity matrix in parallel using a memory-mapped file.
def compute_all_dissim_parallel(patches, block_size=100, percentile=95, max_workers=4, memmap_filename='dissim.npy'):
  N = len(patches)
  # Create a memory-mapped file for the dissimilarity matrix.
  dissim = np.memmap(memmap_filename, dtype='float32', mode='w+', shape=(N, N))

  # Create list of tasks: each task covers a block in the upper triangle (i.e. j >= i)
  tasks = []
  for i_start in range(0, N, block_size):
    i_end = min(N, i_start + block_size)
    for j_start in range(i_start, N, block_size):
      j_end = min(N, j_start + block_size)
      tasks.append((patches, i_start, i_end, j_start, j_end, percentile))

  # Use ProcessPoolExecutor with spawn start method.
  ctx = get_context('spawn')
  with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
    futures = {executor.submit(compute_block, task): task for task in tasks}
    # Use tqdm to track progress.
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing blocks"):
      i_start, i_end, j_start, j_end, block = future.result()
      # Write the block to the memmap matrix
      dissim[i_start:i_end, j_start:j_end] = block
      # If this is an off-diagonal block, fill the symmetric part.
      if i_start != j_start:
        dissim[j_start:j_end, i_start:i_end] = block.T
      dissim.flush()
  return dissim

def get_representations():
  imagenette = False
  coco = False
  largecoco = False
  Rect = False
  BSDS = False
  coco_exp = False
  pascal = False
  pascal_sparse = True
  MINC= False
  MIT = False
  if BSDS:
    # bsdsMask('BSDS500/data/groundTruth/test')
    image, filenames = resnet50_input.test_inputs(FLAGS.data_dir)
    filenames = [element.split('.')[0] for element in filenames]
    # Build a Graph that computes the logits predictions from the
    # inference model.
    _, _, fuse = resnet50.inference(image, tf.zeros([1,image.shape[1]//4+image.shape[1]%4,image.shape[2]//4+image.shape[2]%4,resnet50_input.NUM_CLASSES]), tf.zeros([1,image.shape[1]//4+image.shape[1]%4,image.shape[2]//4+image.shape[2]%4,resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                               intra_op_parallelism_threads=1)) as sess:
      sess.run(tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables()))
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
                                           start=True))

        dir = 'BSDS500/data/images/test'
        dirlist = sorted(os.listdir(dir))

        # pos_dist, neg_dist = getHist(sess, image, GT, fuse)
        # pos_dist, neg_dist = getPatchDS(sess, image, GT, fuse, patch_size=8, ifPCA=True, n=6)
        # pos_d_pca = []
        # neg_d_pca = []
        # for i in range(1,9):
        #   pos_dist, neg_dist = getPatchDS(sess, image, GT, fuse, patch_size=4, ifPCA=True, n=2**i)
        #   pos_d_pca.append([pos_dist])
        #   neg_d_pca.append([neg_dist])
        # pos_dist, neg_dist = getPatchDS(sess, image, GT, fuse, patch_size=4, ifPCA=True, n=16)
        patches = getPatchDS_new(sess, image, fuse, patch_size=8, ifPCA=True, n=16, filenames=filenames, perClass=True)
        with open('patches_8_bsds_{}.pkl'.format(datetime.now()), 'wb') as f:
          pickle.dump(patches, f)
        with open('patches_8_bsds.pkl', 'rb') as f:
          patches = pickle.load(f)
        # dissim = np.zeros((len(patches), len(patches)))
        # doues = comb(patches, 2)
        # with multiprocessing.Pool(processes=16) as p:
        #   with tqdm(total=(len(patches) ** 2)/2) as pbar:
        #     for d in p.imap_unordered(calcPatchDistmulti, doues):
        #       pbar.update()
        #       dissim[d[0], d[1]] = d[2]
        #       dissim[d[1], d[0]] = d[2]
        # # pos_d_pca.append([pos_dist])
        # # neg_d_pca.append([neg_dist])
        #
        # # for i in range(FLAGS.num_examples):
        # #   im, gt, rep = sess.run([image, GT, fuse])
        # #   rep = np.squeeze(rep)
        # #   res_im = np.squeeze(rep)
        # #   res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
        # #   res_reduced = PCA(n_components=3, svd_solver='full').fit_transform(res_vec)
        # #   rep_3 = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], 3))
        # #   rep_3 = rep_3- np.amin(rep_3)/np.amax(rep_3- np.amin(rep_3))
        # #   xVec = np.arange(0, rep.shape[0]-1, 4)
        # #   yVec = np.arange(0, rep.shape[1]-1, 4)
        # #
        # #   # ax2.imshow(im.squeeze())
        # #   # file_name = ''
        # #   # for img in dirlist:
        # #   #   if img.split('.')[1] == 'jpg':
        # #   #     # img_read = plt.imread(dir + '/' + img)/255
        # #   #     # img_read -= np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
        # #   #     img_read = tf.io.read_file(dir + '/' + img)
        # #   #     # image = tf.read_file(queue[0])
        # #   #     img_read = tf.image.decode_jpeg(img_read, channels=3)
        # #   #     try:
        # #   #       img_read.set_shape([321, 481, 3])
        # #   #     except:
        # #   #       img_read.set_shape([481, 321, 3])
        # #   #     img_read = tf.image.convert_image_dtype(img_read, tf.float32)
        # #   #     # bsds_mean = np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
        # #   #     # img_read = img_read - bsds_mean
        # #   #     img_read = sess.run(img_read)
        # #   #     if im.squeeze().shape==img_read.shape:
        # #   #       if np.sum(img_read-im.squeeze())==0:
        # #   #         file_name = img.split('.')[0]
        # #   #         dirlist.remove(img)
        # #   #         break
        # #   # if not file_name:
        # #   #   continue
        # #   for x in xVec:
        # #     for y in yVec:
        # #       # x = np.random.randint(0, rep.shape[1]-1, 2)
        # #       # y = np.random.randint(0, rep.shape[2]-1, 2)
        # #       pix1 = rep[x, y, :]
        # #       randx = np.random.choice(rep.shape[0]-1)
        # #       randy = np.random.choice(rep.shape[1]-1)
        # #       # randx = x+5
        # #       # randy = y+5
        # #       pix2 = rep[randx, randy, :]
        # #       dist = np.sqrt(sum(np.power((pix1 - pix2), 2)))
        # #
        # #       # GT = loadmat(os.sep.join(['BSDS500/data/groundTruth/test/' + filenames[i] + '.mat']))['groundTruth']
        # #       # person1 = GT[0, 0][0, 0][0]
        # #       # person2 = GT[0, 1][0, 0][0]
        # #       # person3 = GT[0, 2][0, 0][0]
        # #       # person4 = GT[0, 3][0, 0][0]
        # #       #
        # #       x_im = x * 4
        # #       y_im = y * 4
        # #       randx_im = randx * 4
        # #       randy_im = randy * 4
        # #       if x_im==randx_im and y_im==randy_im:
        # #         continue
        # #
        # #       score = bool(mostCommonClass(gt, x_im, y_im) == mostCommonClass(gt, randx_im, randy_im))
        # #       # p1_score = int(mostCommonClass(person1, x_im, y_im) == mostCommonClass(person1, randx_im, randy_im))
        # #       # p2_score = int(mostCommonClass(person2, x_im, y_im) == mostCommonClass(person2, randx_im, randy_im))
        # #       # p3_score = int(mostCommonClass(person3, x_im, y_im) == mostCommonClass(person3, randx_im, randy_im))
        # #       # p4_score = int(mostCommonClass(person4, x_im, y_im) == mostCommonClass(person4, randx_im, randy_im))
        # #       # if GT.shape[1]==5:
        # #       #   person5 = GT[0, 4][0, 0][0]
        # #       #   p5_score = int(mostCommonClass(person5, x_im, y_im) == mostCommonClass(person5, randx_im, randy_im))
        # #       #   tot_score = sum((p1_score, p2_score, p3_score, p4_score, p5_score)) / 5
        # #       # else:
        # #       #   tot_score = sum((p1_score, p2_score, p3_score, p4_score)) / 4
        # #       if score:
        # #         pos_dist.append(dist)
        # #       else:
        # #         neg_dist.append(dist)
        # #       # if tot_score>0.7:
        # #       #   pos_dist.append(dist)
        # #       # if tot_score<0.3:
        # #       #   neg_dist.append(dist)
        # #       # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # #       # ax1.imshow(rep_3/np.amax(rep_3))
        # #       # ax3.imshow(person1)
        # #       # ax4.imshow(person2)
        # #       # ax2.imshow(person3)
        # #       # plt.show()
        # #       # plt.close()
        # #       # plt.imshow(rep_3)
        # #       # plt.scatter([y_im, randy_im], [x_im, randx_im], s=30, marker='o', c='r')
        # #       # plt.show()
        # #       # a=1
        # with open('dissim_8_bsds.pkl', 'wb') as f:
        #   pickle.dump(dissim, f)
        with open('dissim_8_bsds.pkl', 'rb') as f:
          dissim = pickle.load(f)
        max_d = np.amax(dissim)
        affinity = 1 - (dissim / max_d)
        sigma = np.percentile(dissim, 1)
        affinity = np.exp(-dissim/sigma)
        D = np.zeros(affinity.shape)
        for i in range(affinity.shape[0]):
          D[i, i] = sum(affinity[i, :])
        L = np.matmul(np.matmul(np.diag(np.diag(D) ** (-1 / 2)), affinity), np.diag(np.diag(D) ** (-1 / 2)))
        eig = np.linalg.eig(L)
        vLens = range(6, 51, 2)
        # vLens = [18]
        clustersCenterD = []
        inClustersD = []
        ns = range(10,101,5)
        # ns = [40]
        fig2, ax2 = plt.subplots()
        for n in ns:
          for vLen in vLens:
            space = eig[1][:, :vLen]
            normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
            Y = (space.T / normalize_f).T
            Kmeans = sklearn.cluster.KMeans(n_clusters=n, algorithm='elkan')
            clusters = Kmeans.fit(Y)
            clusters_trans = Kmeans.fit_transform(Y)
            clustersCenterD.append(np.mean(
              [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
               for j in range(n) if i != j]))
            inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
        # # plot the change in the distances according to the size of the new patches descriptors
        #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
        #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
        #   ax2.set_xlabel('# eig vec')
        #   ax2.set_ylabel('mean distance')
        #   ax2.legend()

            # simulate and test the population of one cluster
            bsdsClasses = np.unique([str(p['label'])+','+str(p['seg']) for p in patches], return_counts=True)
            numclasses = bsdsClasses[0].shape
            idxclasses = [(i, bsdsClasses[0][i]) for i in range(numclasses[0])]
            population = np.zeros((n, 10))  # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
            # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
            clustersNames = range(n)
            # labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
            #                'scissors background', 'airplane', 'airplane background']
            # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
            #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
            counter = np.zeros(numclasses[0])
            for clusterIDX in range(n):
              idxlist = np.where(clusters.labels_ == clusterIDX)
              #       print(len(idxlist[0])/len(patches))
              GT_cluster = [str(p['label'])+','+str(p['seg']) for p in patches if p['idx'] in idxlist[0]]
              classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
              for i, pop in enumerate(classesPop[0]):
                val = [item[0] for item in idxclasses if item[1] == pop][0]
                population[clusterIDX, val] = classesPop[1][i]/len(idxlist[0])
                # counter[val] += 1
            #       for i, pop in enumerate(classesPop[0]):
            #         if b'elephant' in pop and b'255' in pop:
            #           val = 4
            #         if b'elephant' in pop and b'0' in pop:
            #           val = 5
            #         if b'dog' in pop and b'255' in pop:
            #           val = 2
            #         if b'dog' in pop and b'0' in pop:
            #           val = 3
            #         if b'fire hydrant' in pop and b'255' in pop:
            #           val = 6
            #         if b'fire hydrant' in pop and b'0' in pop:
            #           val = 7
            #         if b'train' in pop and b'255' in pop:
            #           val = 8
            #         if b'train' in pop and b'0' in pop:
            #           val = 9
            #         if b'airplane' in pop and b'255' in pop:
            #           val = 0
            #         if b'airplane' in pop and b'0' in pop:
            #           val = 1
            #         classesPop[0][i] = val
            #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
            #         counter[val] += 1
            fig, ax = plt.subplots(figsize=(4, 10))
            # counter = np.broadcast_to(counter, population.shape)
            # population = np.divide(population, counter)
            im = ax.imshow(population)
            ax.set_yticks(np.arange(len(clustersNames)))
            ax.set_xticks(np.arange(numclasses[0]))
            cb = plt.colorbar(im)
            plt.savefig(
              'bsds_heatmaps_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
            plt.close(fig)

        plt.show()
        # plt.savefig('keepDaway_bsds.png')
        # plt.close(fig2)


      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
    # plt.hist(pos_dist, bins=50)
    # plt.hist(neg_dist, bins=50, alpha=0.3)
    # plt.show()
  elif imagenette:
    image, label = getRep_imagenette()
    _, _, fuse = resnet50.inference(image, tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]), tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1)) as sess:
      sess.run(tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables()))
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
                                           start=True))
        patches = getPatchDS_imagenette(sess, image, label, fuse, patch_size=8, ifPCA=True, n=16)

        # with open('patches_8.pkl', 'rb') as f:
        #   patches = pickle.load(f)
        # patches = random.choices(patches, k=1000)
        dissim = np.zeros((len(patches), len(patches)))
        for p1 in tqdm(patches):
            for p2 in patches:
              if dissim[p1['idx'], p2['idx']] != 0:
                continue
              else:
                # start = time.time()
                # patch1 = torch.from_numpy(p1['patch'])
                # patch2 = torch.from_numpy(p2['patch'])
                # dist = calcPatchDisttorch(patch1, patch2)
                dist = calcPatchDist(p1['patch'], p2['patch'])
                dissim[p1['idx'], p2['idx']] = dist
                dissim[p2['idx'], p1['idx']] = dist
                # print(time.time()-start)
        max_d = np.amax(dissim)
        affinity = 1-(dissim/max_d)
        D = np.zeros(affinity.shape)
        for i in range(affinity.shape[0]):
          D[i,i] = sum(affinity[i,:])
        L = np.matmul(np.matmul(np.diag(np.diag(D)**(-1/2)), affinity), np.diag(np.diag(D)**(-1/2)))
        eig = np.linalg.eig(L)
        space = eig[1][:, :10]
        normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
        Y = (space.T / normalize_f).T
        patches = sorted(patches, key=lambda p: p['idx'])
        mymetric = sklearn.metrics.make_scorer(calcPatchDist)
        clustering_kmeans = SpectralClustering(n_clusters=50,
                                        assign_labels='kmeans',
                                        random_state=0,
                                        affinity='nearest_neighbors').fit(np.array([p['patch'].flatten() for p in patches]))
        clustering_disc = SpectralClustering(n_clusters=50,
                                               assign_labels='discretize',
                                               random_state=0,
                                               affinity='nearest_neighbors').fit(
          np.array([p['patch'].flatten() for p in patches]))

        plt.hist(clustering_disc.labels_, bins=50)
        plt.hist(clustering_kmeans.labels_, bins=50)
        a =1




      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  elif largecoco:
    patches = getPatchDS_frompkl()
    with open('patches_largecoco.pkl', 'wb') as f:
      pickle.dump(patches, f)
    # with open('patches_largecoco.pkl', 'rb') as f:
    #   patches = pickle.load(f)
    #       # patches = random.choices(patches, k=500)
    #       # threads_per_block = (16,16)
    #       # blocks_per_grid = ((4224+16-1)//16, (4224+16-1)//16)
    #       # patches_array = [p['patch'] for p in patches]
    #       # patches_cuda = cuda.to_device(patches_array)
    doues = comb(patches, 2)
    #       # dissim = calcPatchDistmulti(doues)
    dissim = np.zeros((len(patches), len(patches)))
    #       # dissim = cuda.to_device(dissim)
    #       # dissim = calcDistGPUPar[blocks_per_grid,threads_per_block](patches_cuda, dissim)
    with multiprocessing.Pool(processes=16) as p:
      with tqdm(total=(len(patches) ** 2) / 2) as pbar:
        for d in p.imap_unordered(calcPatchDistmulti, doues):
          pbar.update()
          dissim[d[0], d[1]] = d[2]
          dissim[d[1], d[0]] = d[2]
    #       # # for p1 in tqdm(patches):
    #       # #   for p2 in patches:
    #       # #     if dissim[p1['idx'], p2['idx']] != 0:
    #       # #       continue
    #       # #     else:
    #       # #       # start = time.time()
    #       # #       # patch1 = torch.from_numpy(p1['patch'])
    #       # #       # patch2 = torch.from_numpy(p2['patch'])
    #       # #       # dist = calcPatchDisttorch(patch1, patch2)
    #       # #       dist = calcPatchDist(p1['patch'], p2['patch'])
    #       # #       dissim[p1['idx'], p2['idx']] = dist
    #       # #       dissim[p2['idx'], p1['idx']] = dist
    #       # #       # print(time.time()-start)
    with open('dissim_largecoco_{}.pkl'.format(datetime.now()), 'wb') as f:
      pickle.dump(dissim, f)
    #       # with open('dissim_8_coco.pkl', 'rb') as f:
    #       #   dissim = pickle.load(f)
    max_d = np.amax(dissim)
    affinity = 1 - (dissim / max_d)
    sigma = np.percentile(dissim, 2)
    affinity = np.exp(-dissim / sigma)
    D = np.zeros(affinity.shape)
    for i in range(affinity.shape[0]):
      D[i, i] = sum(affinity[i, :])
    L = np.matmul(np.matmul(np.diag(np.diag(D) ** (-1 / 2)), affinity), np.diag(np.diag(D) ** (-1 / 2)))
    eig = np.linalg.eig(L)
    vLens = range(5, 81, 5)
    # vLens = [18]
    clustersCenterD = []
    inClustersD = []
    ns = range(20, 301, 20)
    # ns = [40]
    fig2, ax2 = plt.subplots()
    for n in ns:
      for vLen in vLens:
        space = eig[1][:, :vLen]
        normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
        Y = (space.T / normalize_f).T
        Kmeans = sklearn.cluster.KMeans(n_clusters=n, algorithm='elkan')
        clusters = Kmeans.fit(Y)
        clusters_trans = Kmeans.fit_transform(Y)
        clustersCenterD.append(np.mean(
          [[np.linalg.norm((clusters.cluster_centers_[i, :] - clusters.cluster_centers_[j, :]), 2) for i in
            range(n)]
           for j in range(n) if i != j]))
        inClustersD.append(np.mean(
          [np.linalg.norm((Y[i] - clusters.cluster_centers_[clusters.labels_[i]]), 2) for i in
           range(len(patches))]))
      # plot the change in the distances according to the size of the new patches descriptors
      ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
      ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
      ax2.set_xlabel('# eig vec')
      ax2.set_ylabel('mean distance')
      ax2.legend()

    # # simulate and test the population of one cluster
    #   population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
    #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
    #   clustersNames = range(n)
    #   labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background','elephant',
    #   'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
    #   for clusterIDX in range(n):
    #     idxlist = np.where(clusters.labels_ == clusterIDX)
    #     print(len(idxlist)/len(patches))
    #     GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
    #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
    #     for i, pop in enumerate(classesPop[0]):
    #       if b'airplane' in pop and b'255' in pop:
    #         classesPop[0][i] = 0
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'airplane' in pop and b'0' in pop:
    #         classesPop[0][i] = 1
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'dog' in pop and b'255' in pop:
    #         classesPop[0][i] = 2
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'dog' in pop and b'0' in pop:
    #         classesPop[0][i] = 3
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'fire hydrant' in pop and b'255' in pop:
    #         classesPop[0][i] = 6
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'fire hydrant' in pop and b'0' in pop:
    #         classesPop[0][i] = 7
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'elephant' in pop and b'255' in pop:
    #         classesPop[0][i] = 4
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'elephant' in pop and b'0' in pop:
    #         classesPop[0][i] = 5
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'train' in pop and b'255' in pop:
    #         classesPop[0][i] = 8
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #       if b'train' in pop and b'0' in pop:
    #         classesPop[0][i] = 9
    #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    #   fig, ax = plt.subplots()
    #   im = ax.imshow(population)
    #   ax.set_yticks(np.arange(len(clustersNames)))
    #   ax.set_xticks(np.arange(len(labelsNames)))
    #   cb = plt.colorbar(im)
    #   plt.savefig('heatmaps_coco_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
    #   plt.close(fig)
    # population[clusterIDX, classesPop[0]] = classesPop[1]
    # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
    #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])
    plt.show()
    plt.savefig('keepDaway_newpatches.png')
    plt.close(fig2)
    fig, ax = plt.subplots()
    ax.imshow(population)
    ax.set_yticks(np.arange(len(clustersNames)), labels=clustersNames)
    ax.set_xticks(np.arange(len(labelsNames)), labels=labelsNames)
    ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
    random.shuffle(ims)
    fig, axs = plt.subplots(6, 6)
    for i in range(6):
      for j in range(6):
        axs[i, j].imshow(ims[i * 6 + j])
    plt.show()
    # patches = sorted(patches, key=lambda p: p['idx'])
    # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
    # clustering_kmeans = SpectralClustering(n_clusters=50,
    #                                        assign_labels='kmeans',
    #                                        random_state=0,
    #                                        affinity='nearest_neighbors').fit(
    #   np.array([p['patch'].flatten() for p in patches]))
    # clustering_disc = SpectralClustering(n_clusters=50,
    #                                      assign_labels='discretize',
    #                                      random_state=0,
    #                                      affinity='nearest_neighbors').fit(
    #   np.array([p['patch'].flatten() for p in patches]))
    #
    # plt.hist(clustering_disc.labels_, bins=50)
    # plt.hist(clustering_kmeans.labels_, bins=50)
  # elif coco:
  #   image, mask, label = getRep_coco()
  #   _, _, fuse = resnet50.inference(image, tf.zeros(
  #     [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
  #      resnet50_input.NUM_CLASSES]), tf.zeros(
  #     [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
  #      resnet50_input.NUM_CLASSES]))
  #   saver = tf.train.Saver(tf.global_variables())
  #   with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
  #                                         intra_op_parallelism_threads=1)) as sess:
  #     sess.run(tf.group(tf.initialize_all_variables(),
  #                       tf.initialize_local_variables()))
  #     ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  #     if ckpt and ckpt.model_checkpoint_path:
  #       saver.restore(sess, ckpt.model_checkpoint_path)
  #     else:
  #       print('No checkpoint file found')
  #       return
  #     # Start the queue runners.
  #     coord = tf.train.Coordinator()
  #     try:
  #       threads = []
  #       for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
  #         threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
  #                                          start=True))
  #       patches = getPatchDS_COCO_new(sess, image, mask, label, fuse, patch_size=8, ifPCA=True, n=24)
  #       # patches = getPatchDS_frompkl()
  #
  #       with open('patches_8_coco_pca24.pkl', 'wb') as f:
  #         pickle.dump(patches, f)
  #       with open('patches_8_coco_pca24.pkl', 'rb') as f:
  #         patches = pickle.load(f)
  #       #       # patches = random.choices(patches, k=500)
  #       #       # threads_per_block = (16,16)
  #       #       # blocks_per_grid = ((4224+16-1)//16, (4224+16-1)//16)
  #       #       # patches_array = [p['patch'] for p in patches]
  #       #       # patches_cuda = cuda.to_device(patches_array)
  #       doues = comb(patches, 2)
  #       #       # dissim = calcPatchDistmulti(doues)
  #       dissim = np.zeros((len(patches), len(patches)))
  #       #       # dissim = cuda.to_device(dissim)
  #       #       # dissim = calcDistGPUPar[blocks_per_grid,threads_per_block](patches_cuda, dissim)
  #       with multiprocessing.Pool(processes=16) as p:
  #         with tqdm(total=(len(patches) ** 2) / 2) as pbar:
  #           for d in p.imap_unordered(calcPatchDistmulti, doues):
  #             pbar.update()
  #             dissim[d[0], d[1]] = d[2]
  #             dissim[d[1], d[0]] = d[2]
  #       #       # # for p1 in tqdm(patches):
  #       #       # #   for p2 in patches:
  #       #       # #     if dissim[p1['idx'], p2['idx']] != 0:
  #       #       # #       continue
  #       #       # #     else:
  #       #       # #       # start = time.time()
  #       #       # #       # patch1 = torch.from_numpy(p1['patch'])
  #       #       # #       # patch2 = torch.from_numpy(p2['patch'])
  #       #       # #       # dist = calcPatchDisttorch(patch1, patch2)
  #       #       # #       dist = calcPatchDist(p1['patch'], p2['patch'])
  #       #       # #       dissim[p1['idx'], p2['idx']] = dist
  #       #       # #       dissim[p2['idx'], p1['idx']] = dist
  #       #       # #       # print(time.time()-start)
  #       with open('dissim_8_coco_new_pca24_{}.pkl'.format(datetime.now()), 'wb') as f:
  #         pickle.dump(dissim, f)
  #       #       # with open('dissim_8_coco.pkl', 'rb') as f:
  #       #       #   dissim = pickle.load(f)
  #       max_d = np.amax(dissim)
  #       affinity = 1 - (dissim / max_d)
  #       sigma = np.percentile(dissim, 85)
  #       affinity = np.exp(-dissim / sigma)
  #       D = np.zeros(affinity.shape)
  #       for i in range(affinity.shape[0]):
  #         D[i, i] = sum(affinity[i, :])
  #       L = np.matmul(np.matmul(np.diag(np.diag(D) ** (-1 / 2)), affinity), np.diag(np.diag(D) ** (-1 / 2)))
  #       eig = np.linalg.eig(L)
  #       vLens = range(6, 51, 2)
  #       # vLens = [18]
  #       clustersCenterD = []
  #       inClustersD = []
  #       ns = range(10, 51, 5)
  #       # ns = [40]
  #       fig2, ax2 = plt.subplots()
  #       for n in ns:
  #         for vLen in vLens:
  #           space = eig[1][:, :vLen]
  #           normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
  #           Y = (space.T / normalize_f).T
  #           Kmeans = sklearn.cluster.KMeans(n_clusters=n, algorithm='elkan')
  #           clusters = Kmeans.fit(Y)
  #           clusters_trans = Kmeans.fit_transform(Y)
  #           clustersCenterD.append(np.mean(
  #             [[np.linalg.norm((clusters.cluster_centers_[i, :] - clusters.cluster_centers_[j, :]), 2) for i in
  #               range(n)]
  #              for j in range(n) if i != j]))
  #           inClustersD.append(np.mean(
  #             [np.linalg.norm((Y[i] - clusters.cluster_centers_[clusters.labels_[i]]), 2) for i in
  #              range(len(patches))]))
  #         # plot the change in the distances according to the size of the new patches descriptors
  #         ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
  #         ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
  #         ax2.set_xlabel('# eig vec')
  #         ax2.set_ylabel('mean distance')
  #         ax2.legend()
  #
  #       # # simulate and test the population of one cluster
  #       #   population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
  #       #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
  #       #   clustersNames = range(n)
  #       #   labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background','elephant',
  #       #   'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
  #       #   for clusterIDX in range(n):
  #       #     idxlist = np.where(clusters.labels_ == clusterIDX)
  #       #     print(len(idxlist)/len(patches))
  #       #     GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
  #       #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
  #       #     for i, pop in enumerate(classesPop[0]):
  #       #       if b'airplane' in pop and b'255' in pop:
  #       #         classesPop[0][i] = 0
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'airplane' in pop and b'0' in pop:
  #       #         classesPop[0][i] = 1
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'dog' in pop and b'255' in pop:
  #       #         classesPop[0][i] = 2
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'dog' in pop and b'0' in pop:
  #       #         classesPop[0][i] = 3
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'fire hydrant' in pop and b'255' in pop:
  #       #         classesPop[0][i] = 6
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'fire hydrant' in pop and b'0' in pop:
  #       #         classesPop[0][i] = 7
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'elephant' in pop and b'255' in pop:
  #       #         classesPop[0][i] = 4
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'elephant' in pop and b'0' in pop:
  #       #         classesPop[0][i] = 5
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'train' in pop and b'255' in pop:
  #       #         classesPop[0][i] = 8
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #       if b'train' in pop and b'0' in pop:
  #       #         classesPop[0][i] = 9
  #       #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
  #       #   fig, ax = plt.subplots()
  #       #   im = ax.imshow(population)
  #       #   ax.set_yticks(np.arange(len(clustersNames)))
  #       #   ax.set_xticks(np.arange(len(labelsNames)))
  #       #   cb = plt.colorbar(im)
  #       #   plt.savefig('heatmaps_coco_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
  #       #   plt.close(fig)
  #       # population[clusterIDX, classesPop[0]] = classesPop[1]
  #       # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
  #       #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])
  #       plt.show()
  #       plt.savefig('keepDaway_newpatches.png')
  #       plt.close(fig2)
  #       fig, ax = plt.subplots()
  #       ax.imshow(population)
  #       ax.set_yticks(np.arange(len(clustersNames)), labels=clustersNames)
  #       ax.set_xticks(np.arange(len(labelsNames)), labels=labelsNames)
  #       ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
  #       random.shuffle(ims)
  #       fig, axs = plt.subplots(6, 6)
  #       for i in range(6):
  #         for j in range(6):
  #           axs[i, j].imshow(ims[i * 6 + j])
  #       plt.show()
  #       # patches = sorted(patches, key=lambda p: p['idx'])
  #       # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
  #       # clustering_kmeans = SpectralClustering(n_clusters=50,
  #       #                                        assign_labels='kmeans',
  #       #                                        random_state=0,
  #       #                                        affinity='nearest_neighbors').fit(
  #       #   np.array([p['patch'].flatten() for p in patches]))
  #       # clustering_disc = SpectralClustering(n_clusters=50,
  #       #                                      assign_labels='discretize',
  #       #                                      random_state=0,
  #       #                                      affinity='nearest_neighbors').fit(
  #       #   np.array([p['patch'].flatten() for p in patches]))
  #       #
  #       # plt.hist(clustering_disc.labels_, bins=50)
  #       # plt.hist(clustering_kmeans.labels_, bins=50)
  #
  #
  #     except Exception as e:  # pylint: disable=broad-except
  #       coord.request_stop(e)
  #
  #     coord.request_stop()
  #     coord.join(threads, stop_grace_period_secs=10)
  elif coco_exp:
    image,mask, label = getRep_coco()
    _, _, fuse = resnet50.inference(image, tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]), tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1)) as sess:
      model_dict = np.load('fp_weights.npy', allow_pickle=True).item()
      all_vars = tf.trainable_variables()
      for v in all_vars:
        if (v.op.name.find("weights") > -1) and (v.op.name.find("softmax_linear") == -1) and (
                v.op.name.find("fuse") == -1) and (v.op.name.find("conv5_3") == -1):
          assign_op = v.assign(model_dict[v.op.name])
          sess.run(assign_op)

      sess.run(tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables()))
      # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      # if ckpt and ckpt.model_checkpoint_path:
      #   saver.restore(sess, ckpt.model_checkpoint_path)
      # else:
      #   print('No checkpoint file found')
      #   return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
                                           start=True))
        patches = getPatchDS_pascal(sess, image, mask, label, fuse, patch_size=8, ifPCA=True, n=3)
        # patches = getPatchDS_frompkl()

        with open('patches_8_coco_pca24_pascal.pkl', 'wb') as f:
          pickle.dump(patches, f)
        with open('patches_8_coco_pca24_pascal.pkl', 'rb') as f:
          patches = pickle.load(f)
  #       # patches = random.choices(patches, k=500)
  #       # threads_per_block = (16,16)
  #       # blocks_per_grid = ((4224+16-1)//16, (4224+16-1)//16)
  #       # patches_array = [p['patch'] for p in patches]
  #       # patches_cuda = cuda.to_device(patches_array)
        doues = comb(patches, 2)
  #       # dissim = calcPatchDistmulti(doues)
        dissim = np.zeros((len(patches), len(patches)))
  #       # dissim = cuda.to_device(dissim)
  #       # dissim = calcDistGPUPar[blocks_per_grid,threads_per_block](patches_cuda, dissim)
        with multiprocessing.Pool(processes=16) as p:
          with tqdm(total=(len(patches) ** 2)/2) as pbar:
            for d in p.imap_unordered(calcPatchDistmulti, doues):
              pbar.update()
              dissim[d[0], d[1]] = d[2]
              dissim[d[1], d[0]] = d[2]
  #       # # for p1 in tqdm(patches):
  #       # #   for p2 in patches:
  #       # #     if dissim[p1['idx'], p2['idx']] != 0:
  #       # #       continue
  #       # #     else:
  #       # #       # start = time.time()
  #       # #       # patch1 = torch.from_numpy(p1['patch'])
  #       # #       # patch2 = torch.from_numpy(p2['patch'])
  #       # #       # dist = calcPatchDisttorch(patch1, patch2)
  #       # #       dist = calcPatchDist(p1['patch'], p2['patch'])
  #       # #       dissim[p1['idx'], p2['idx']] = dist
  #       # #       dissim[p2['idx'], p1['idx']] = dist
  #       # #       # print(time.time()-start)
        with open('dissim_8_coco_new_pca24_{}.pkl'.format(datetime.now()), 'wb') as f:
          pickle.dump(dissim, f)
  #       # with open('dissim_8_coco.pkl', 'rb') as f:
  #       #   dissim = pickle.load(f)
        max_d = np.amax(dissim)
        affinity = 1 - (dissim / max_d)
        sigma = np.percentile(dissim, 85)
        affinity = np.exp(-dissim/sigma)
        D = np.zeros(affinity.shape)
        for i in range(affinity.shape[0]):
          D[i, i] = sum(affinity[i, :])
        L = np.matmul(np.matmul(np.diag(np.diag(D) ** (-1 / 2)), affinity), np.diag(np.diag(D) ** (-1 / 2)))
        eig = np.linalg.eig(L)
        vLens = range(6, 51, 2)
        # vLens = [18]
        clustersCenterD = []
        inClustersD = []
        ns = range(10,51,5)
        # ns = [40]
        fig2, ax2 = plt.subplots()
        for n in ns:
          for vLen in vLens:
            space = eig[1][:, :vLen]
            normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
            Y = (space.T / normalize_f).T
            Kmeans = sklearn.cluster.KMeans(n_clusters=n, algorithm='elkan')
            clusters = Kmeans.fit(Y)
            clusters_trans = Kmeans.fit_transform(Y)
            clustersCenterD.append(np.mean(
              [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
               for j in range(n) if i != j]))
            inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
        # plot the change in the distances according to the size of the new patches descriptors
          ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
          ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
          ax2.set_xlabel('# eig vec')
          ax2.set_ylabel('mean distance')
          ax2.legend()

        # # simulate and test the population of one cluster
        #   population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
        #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
        #   clustersNames = range(n)
        #   labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background','elephant',
        #   'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
        #   for clusterIDX in range(n):
        #     idxlist = np.where(clusters.labels_ == clusterIDX)
        #     print(len(idxlist)/len(patches))
        #     GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
        #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
        #     for i, pop in enumerate(classesPop[0]):
        #       if b'airplane' in pop and b'255' in pop:
        #         classesPop[0][i] = 0
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'airplane' in pop and b'0' in pop:
        #         classesPop[0][i] = 1
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'dog' in pop and b'255' in pop:
        #         classesPop[0][i] = 2
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'dog' in pop and b'0' in pop:
        #         classesPop[0][i] = 3
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'fire hydrant' in pop and b'255' in pop:
        #         classesPop[0][i] = 6
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'fire hydrant' in pop and b'0' in pop:
        #         classesPop[0][i] = 7
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'elephant' in pop and b'255' in pop:
        #         classesPop[0][i] = 4
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'elephant' in pop and b'0' in pop:
        #         classesPop[0][i] = 5
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'train' in pop and b'255' in pop:
        #         classesPop[0][i] = 8
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #       if b'train' in pop and b'0' in pop:
        #         classesPop[0][i] = 9
        #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #   fig, ax = plt.subplots()
        #   im = ax.imshow(population)
        #   ax.set_yticks(np.arange(len(clustersNames)))
        #   ax.set_xticks(np.arange(len(labelsNames)))
        #   cb = plt.colorbar(im)
        #   plt.savefig('heatmaps_coco_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
        #   plt.close(fig)
          # population[clusterIDX, classesPop[0]] = classesPop[1]
          # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
          #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])
        plt.show()
        plt.savefig('keepDaway_newpatches.png')
        plt.close(fig2)
        fig, ax = plt.subplots()
        ax.imshow(population)
        ax.set_yticks(np.arange(len(clustersNames)), labels=clustersNames)
        ax.set_xticks(np.arange(len(labelsNames)), labels=labelsNames)
        ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
        random.shuffle(ims)
        fig, axs = plt.subplots(6, 6)
        for i in range(6):
          for j in range(6):
            axs[i, j].imshow(ims[i * 6 + j])
        plt.show()
        # patches = sorted(patches, key=lambda p: p['idx'])
        # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
        # clustering_kmeans = SpectralClustering(n_clusters=50,
        #                                        assign_labels='kmeans',
        #                                        random_state=0,
        #                                        affinity='nearest_neighbors').fit(
        #   np.array([p['patch'].flatten() for p in patches]))
        # clustering_disc = SpectralClustering(n_clusters=50,
        #                                      assign_labels='discretize',
        #                                      random_state=0,
        #                                      affinity='nearest_neighbors').fit(
        #   np.array([p['patch'].flatten() for p in patches]))
        #
        # plt.hist(clustering_disc.labels_, bins=50)
        # plt.hist(clustering_kmeans.labels_, bins=50)


      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  elif pascal:
    # image, mask, imname = getRep_pascal()

    # _, _, fuse = resnet50.inference(image, tf.zeros(
    #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    #    resnet50_input.NUM_CLASSES]), tf.zeros(
    #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    #    resnet50_input.NUM_CLASSES]))
    # saver = tf.train.Saver(tf.global_variables())
    # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
    #                                       intra_op_parallelism_threads=1)) as sess:
    #   sess.run(tf.group(tf.initialize_all_variables(),
    #                     tf.initialize_local_variables()))
    #   ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    #   if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #   else:
    #     print('No checkpoint file found')
    #     return
    #   Start the queue runners.
      # coord = tf.train.Coordinator()
    # #
    # # _, _, fuse = resnet50.inference(image, tf.zeros(
    # #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    # #    resnet50_input.NUM_CLASSES]), tf.zeros(
    # #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    # #    resnet50_input.NUM_CLASSES]))
    # # # ### bsds model
    # # saver = tf.train.Saver(tf.global_variables())
    # # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
    # #                                       intra_op_parallelism_threads=1)) as sess:
    # #   sess.run(tf.group(tf.initialize_all_variables(),
    # #                     tf.initialize_local_variables()))
    # #   ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # #   if ckpt and ckpt.model_checkpoint_path:
    # #     saver.restore(sess, ckpt.model_checkpoint_path)
    # #   else:
    # #     print('No checkpoint file found')
    # #     return
    # # # ### pascal weights
    # # # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
    # # #                                       intra_op_parallelism_threads=1)) as sess:
    # # #   model_dict = np.load('fp_weights.npy', allow_pickle=True).item()
    # # #   all_vars = tf.trainable_variables()
    # # #   for v in all_vars:
    # # #     if (v.op.name.find("weights") > -1) and (v.op.name.find("softmax_linear") == -1) and (
    # # #             v.op.name.find("fuse") == -1) and (v.op.name.find("conv5_3") == -1):
    # # #       assign_op = v.assign(model_dict[v.op.name])
    # # #       sess.run(assign_op)
    # # #
    # #   sess.run(tf.group(tf.initialize_all_variables(),
    # #                     tf.initialize_local_variables()))
    # # #   # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # # #   # if ckpt and ckpt.model_checkpoint_path:
    # # #   #   saver.restore(sess, ckpt.model_checkpoint_path)
    # # #   # else:
    # # #   #   print('No checkpoint file found')
    # # #   #   return
    # # #   # Start the queue runners.
    # #   coord = tf.train.Coordinator()
    #   try:
    #     threads = []
    #     for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #       threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
    #                                        start=True))
    #     patches = getPatchDS_pascal(sess, image, mask, imname, fuse, patch_size=8, ifPCA=True, n=16, BG=False)
        # patches = getPatchDS_frompkl()
        #
        # with open('patches_8_pascal_pca16_balanced_noBG_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(patches, f)

        with open('patches_8_pascal_pca16_balanced_noBG_2025-02-02 12:34:41.083454.pkl', 'rb') as f:
          patches = pickle.load(f)
        # with open('patches_8_pascal_pca16_balanced_noBG_2024-12-15 13:06:11.715102.pkl', 'rb') as f:
        #   patches = pickle.load(f)
        # filtered_patches = []
        # remain_patches = patches[:10000]
        # while remain_patches:
        #   p = remain_patches[0]
        #   centroids, ids = getCentroid(p, remain_patches)
        #   for c in centroids:
        #     c['temp_idx']=len(filtered_patches)
        #     filtered_patches.append(c)
        #   remain_patches = remain_patches[len(ids):]
        # dissim = compute_all_dissim_parallel(patches, block_size=5000, percentile=10, max_workers=8)
        # N = len(patches)
        # Open the same raw file in read-only or read-write mode:
        # dissim = create_dissim_sparse(patches)
        # print(dissim.shape)
        # scores = dist_methods_comparison(patches)
        # with open('patches_8_pascal_pca16_balanced_noBG_2024-03-07 12:22:39.858088.pkl', 'rb') as f:
        #   patches = pickle.load(f)
        # classes2inc = [82, 99, 123]
        # patches = [p for p in patches if p['seg'] in classes2inc]
        # for i in range(len(patches)):
        #   patches[i]['idx']=i
        # new_patches = random.sample(patches, 1800)
        # with open('patches_8_pascal_pca16_filtered_sampled.pkl', 'rb') as f:
        #   new_patches = pickle.load(f)
        # lda = LDA(new_patches)
  #     # dissim =
  # #
  # # # # #       # patches = random.choices(patches, k=500)
  # # # # #       # threads_per_block = (16,16)
  # # # # #       # blocks_per_grid = ((4224+16-1)//16, (4224+16-1)//16)
  # # # # #       # patches_array = [p['patch'] for p in patches]
  # # # # #       # patches_cuda = cuda.to_device(patches_array)
  #       doues = comb(new_patches, 2)
  # #       # doues = [(d, lda) for d in doues]
  # # # # # # # # #       # dissim = calcPatchDistmulti(doues)
  #       data = []
  #       labels = []
  # #       # p1order = []
  # #       # p2order = []
  #       with multiprocessing.Pool(processes=16) as p:
  #           with tqdm(total=(len(new_patches) ** 2)/2) as pbar:
  #             for out in p.imap_unordered(calcPatchDist4lda, doues):
  #               pbar.update()
  #               _, _, d, l = out
  #               # p1 = np.where([p['idx']==p1 for p in new_patches])
  #               # p2 = np.where([p['idx']==p2 for p in new_patches])
  #               # p1order.append(p1)
  #               # p2order.append(p2)
  #               data.append(d)
  #               labels.append(int(l))
  #       data = np.array(data)
  #       labels = np.array(labels)
  #       weights = lda_fromScratch(data, labels)
  #       patches = [p for p in patches if p['seg']!=np.uint8(30)]
  #       with open('Thesis figures/dissim_8_pascal_noBG_q10_fullDS.pkl', 'rb') as f:
          # with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
          #   dissim = pickle.load(mmapped_file)
          # dissim = pickle.load(f)
        # dissim = pd.read_pickle('Thesis figures/dissim_8_pascal_noBG_q10_fullDS.pkl')
        # patches = patches[:1000]
        # doues = itertools.product(patches[11000:15000], repeat=2)
        # dissim = np.zeros((len(patches), len(patches)))

        # for i in tqdm(range(len(patches))):
        #   combinations = list(itertools.product([patches[i]], patches))
        # combinations_dict = {first: [(first, second) for (first, second) in combinations if first == pair[0]] for pair in
        #                combinations}

        # dissim = np.load('dissim_8_pascal_noBG_q10_fullDS.npz')['arr_0']
        # doues = [d for d in doues if dissim[d[0]['idx'], d[1]['idx']] == 0]
        # dists = []
  #       with multiprocessing.Pool(processes=16) as p:
  #         with tqdm(total=(len(patches) ** 2) / 2) as pbar:
  #           for d in p.imap_unordered(calcPatchDist4lda, doues):
  #             pbar.update()
  #             p1, p2, dist, l = d
              # dissim[p1,p2] = dissim[p2,p1] = np.array(dist).dot(weights)
        # with multiprocessing.Pool(processes=16) as p:
        #     with tqdm(total=(len(patches) ** 2)/2) as pbar:
        #       for d in p.imap_unordered(transorm2LDA, doues):
        #         pbar.update()
        #         dissim[d[0], d[1]] = d[2]
        #         dissim[d[1], d[0]] = d[2]
  # # # # #       # dissim = cuda.to_device(dissim)
  # # # # #       # dissim = calcDistGPUPar[blocks_per_grid,threads_per_block](patches_cuda, dissim)
  #       counter = 11000
  #       dists = np.zeros(4000)
  #       with multiprocessing.Pool(processes=16) as p:
  #         with tqdm(total=(len(patches) ** 2)) as pbar:
  #           for d in p.map(calcPatchDistmulti, doues):
  #             pbar.update()
  # #             dists[d[1]-11000] = d[2]
  # #             if d[1]==14999:#len(patches)-1:
  # #               perc = np.percentile(dists, 3)
  # #               idxs = np.where(dists<=perc) #np.logical_and(,dists > 0))
  # #               for idx in idxs[0]:
  # #                 dissim[counter, idx+11000] = dists[idx]
  # #                 dissim[idx+11000, counter] = dists[idx]
  # #               counter += 1
  # #               dists = np.zeros(4000)
  #               # dists = []
  #               dissim[d[0], d[1]] = d[2]
  #               dissim[d[1], d[0]] = d[2]
        # try:

            # if pbar.n % 100000000==0:
            #   np.savez_compressed('dissim_8_pascal_noBG_q10_fullDS.npz', dissim)
            # with open('dissim_8_pascal_noBG_q10_fullDS.pkl', 'wb') as f:
            #   pickle.dump(dissim, f, protocol=4)
        # np.savez_compressed('dissim_8_pascal_noBG_q10_fullDS.npz', dissim)
        # except:
        #   continue
        #       # dists.append([d[3], (d[0], d[1])])
  # # # #       # # for p1 in tqdm(patches):
  # # # #       # #   for p2 in patches:
  # # # #       # #     if dissim[p1['idx'], p2['idx']] != 0:
  # # # #       # #       continue
  # # # #       # #     else:
  # # # #       # #       # start = time.time()
  # # # #       # #       # patch1 = torch.from_numpy(p1['patch'])
  # # # #       # #       # patch2 = torch.from_numpy(p2['patch'])
  # # # #       # #       # dist = calcPatchDisttorch(patch1, patch2)
  # # # #       # #       dist = calcPatchDist(p1['patch'], p2['patch'])
  # # # #       # #       dissim[p1['idx'], p2['idx']] = dist
  # # # #       # #       dissim[p2['idx'], p1['idx']] = dist
  # # # #       # #       # print(time.time()-start)
  #       dissim = dissim[np.any(dissim != 0, axis=1), :][:, np.any(dissim != 0, axis=1)]


        # with open('dissim_8_pascal_noBG_q10_balanced_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(dissim, f)
        with open('dissim_8_pascal_noBG_q10_balanced_2025-01-20 13:08:39.986150.pkl', 'rb') as f:
          dissim = pickle.load(f)
        # with open('dissim_8_minipascal_avg_filtered_q0_noBG_2024-02-07 11:26:37.549013.pkl', 'rb') as f:
        #   dissim = pickle.load(f)
        # with open('dissim_8_pascal_avg_filtered_q0_noBG_balanced_2024-03-07 13:00:44.615576.pkl', 'rb') as f:
        #   dissim = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q50_2023-09-28 14:59:31.762212.pkl', 'rb') as f:
  #         dissim_q50 = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q80_2023-09-27 16:45:36.574020.pkl', 'rb') as f:
  #         dissim_q80 = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q100_2023-09-27 17:38:53.305407.pkl', 'rb') as f:
  #         dissim_q100 = pickle.load(f)
        max_d = np.amax(dissim)
        # affinity = 1 - (dissim / max_d)
        sigma = np.percentile(dissim, 1)
        # dissim[dissim>np.percentile(dissim, 10)]=np.amax(dissim)
        affinity = np.exp(-dissim/sigma)

        D = np.zeros(affinity.shape)
        for i in range(affinity.shape[0]):
          D[i, i] = sum(affinity[i, :])
        L = D- affinity
        # D_vec = np.diag(D)  # shape (N,)
        # D_inv_sqrt = 1.0 / np.sqrt(D_vec + 1e-10)  # regularization for stability

        # Compute L_sym efficiently:
        # L_sym = L * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
        L_sym = (np.diag(np.diag(D) ** (-1 / 2)) @ L) @ np.diag(np.diag(D) ** (-1 / 2))
        eig = scipy.linalg.eig(L_sym)
        idx = eig[0].argsort()
        eigenValues = eig[0][idx]
        eigenVectors = eig[1][:, idx]
        # with open('eig_balanced_pascal_pca16_filtered_noBG_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(eig, f)
        # vLens = range(10, 151, 10)
        vLens = [20]
        clustersCenterD = []
        inClustersD = []
        # ns = range(50,201,30)
        ns = [150]
        # fig2, ax2 = plt.subplots()
        segvals = [(p['seg']) for p in patches]
        classesPopDS = np.unique(segvals, return_counts=True, axis=0)
        classesIdx = {i: classesPopDS[0][i] for i in range(classesPopDS[0].shape[0])}
        MIs = {}
        conHs = {}
        condHperC = {}
        purity = np.arange(0.1, 1, 0.1)
        chance2cl = {}
        coverage = {}
        for n in tqdm(ns):
          MI_n = []
        #   ims_true = []
        #   ims_false = []
        #   space = eig[1][:, :n]
        #   normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
        #   Y = (space.T / normalize_f).T
        #
        #   Kmeans = sklearn.cluster.KMeans(n_clusters=n, n_init=500)
        #   clusters = Kmeans.fit(Y)
        #   distsFromClustersCenter = clusters.transform(Y)
        #   distsperCluster = [
        #     (distsFromClustersCenter[np.where(clusters.labels_ == i)], np.where(clusters.labels_ == i)) for i in
        #     range(n)]
        #   patches2Include = [distsperCluster[i][1][0][np.where(
        #     distsperCluster[i][0][:, i] <= np.percentile(distsperCluster[i][0][:, i], 100))[0]] for i in range(n)]
        #   # clusters_trans = Kmeans.fit_transform(Y)
        #   # clustersCenterD.append(np.mean(
        #   #   [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
        #   #    for j in range(n) if i != j]))
        #   # inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
        # # # plot the change in the distances according to the size of the new patches descriptors
        # #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
        # #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
        # #   ax2.set_xlabel('# eig vec')
        # #   ax2.set_ylabel('mean distance')
        # #   ax2.legend()
        #   # simulate and test the population of one cluster
        #   population = np.zeros((n, len(classesPopDS[0])))
        #   probs = np.zeros(population.shape)
        #   # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
        #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
        #   clustersNames = range(n)
        #   # labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
        #   #                'scissors background', 'airplane', 'airplane background']
        #   # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
        #   #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
        #   counter = np.zeros((1, n))
        #   for clusterIDX in range(n):
        #     # idxlist = np.where(clusters.labels_ == clusterIDX)
        #     idxlist = [patches2Include[clusterIDX]]
        #     #       print(len(idxlist[0])/len(patches))
        #     GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
        #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
        #     max_Pclass = copy.deepcopy(classesPop[0])[np.argmax(classesPop[1])]
        #     for i, pop in enumerate(classesPop[0]):
        #       val = list(classesIdx.values()).index(pop)
        #       # population[clusterIDX, val] = classesPop[1][i] / len(idxlist[0])
        #       # if b'bus' in pop and b'255' in pop:
        #       #   val = 0
        #       # if b'bus' in pop and b'0' in pop:
        #       #   val = 1
        #       # if b'dog' in pop and b'255' in pop:
        #       #   val = 2
        #       # if b'dog' in pop and b'0' in pop:
        #       #   val = 3
        #       # if b'pizza' in pop and b'255' in pop:
        #       #   val = 4
        #       # if b'pizza' in pop and b'0' in pop:
        #       #   val = 5
        #       # if b'scissors' in pop and b'255' in pop:
        #       #   val = 6
        #       # if b'scissors' in pop and b'0' in pop:
        #       #   val = 7
        #       # if b'airplane' in pop and b'255' in pop:
        #       #   val = 8
        #       # if b'airplane' in pop and b'0' in pop:
        #       #   val = 9
        #       classesPop[0][i] = val
        #       population[clusterIDX, int(classesPop[0][i])] = classesPop[1][i] / sum(classesPop[1])
        #       probs[clusterIDX, int(classesPop[0][i])] = classesPop[1][i]/len(patches)
        #     # max_Pclass = classesPop[0][np.argmax(classesPop[1])]
        #     try:
        #       true_im = [p['ref'] for p in patches if
        #                  (p['idx'] in idxlist[0] and int(max_Pclass) == p['seg'])]
        #       ims_true.append(random.sample(true_im, k=5))
        #     except:
        #       print(max_Pclass)
        #     false_im = [p['ref'] for p in patches if (
        #               p['idx'] in idxlist[0] and int(max_Pclass) != p['seg'])]
        #     try:
        #       ims_false.append(random.sample(false_im, k=5))
        #     except:
        #       ims_false.append(false_im)
        #     counter[0, clusterIDX] = np.sum(classesPop[1])
        #   # #       for i, pop in enumerate(classesPop[0]):
        #   # #         if b'elephant' in pop and b'255' in pop:
        #   # #           val = 4
        #   # #         if b'elephant' in pop and b'0' in pop:
        #   # #           val = 5
        #   # #         if b'dog' in pop and b'255' in pop:
        #   # #           val = 2
        #   # #         if b'dog' in pop and b'0' in pop:
        #   # #           val = 3
        #   # #         if b'fire hydrant' in pop and b'255' in pop:
        #   # #           val = 6
        #   # #         if b'fire hydrant' in pop and b'0' in pop:
        #   # #           val = 7
        #   # #         if b'train' in pop and b'255' in pop:
        #   # #           val = 8
        #   # #         if b'train' in pop and b'0' in pop:
        #   # #           val = 9
        #   # #         if b'airplane' in pop and b'255' in pop:
        #   # #           val = 0
        #   # #         if b'airplane' in pop and b'0' in pop:
        #   # #           val = 1
        #   # #         classesPop[0][i] = val
        #   # #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #   # #         counter[val] += 1
        #   MI1 = sklearn.metrics.mutual_info_score([p['seg'] for p in patches if p['idx'] in np.hstack(patches2Include)], [clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)])
        #   MI_n.append(MI1)
        #   # MI2 = 0
        #   # for i in range(probs.shape[0]):
        #   #   for j in range(probs.shape[1]):
        #   #     if (probs[i, j] * sum(probs[:, j]) * sum(probs[i, :])) != 0:
        #   #       MI2 += probs[i, j] * np.log(probs[i, j] / (sum(probs[:, j]) * sum(probs[i, :])))
        #
        #   # fig, ax = plt.subplots(figsize=(6, 20))
        #   # # counter = np.broadcast_to(counter.T, population.shape)
        #   # # population = np.divide(population, counter)
        #   # im = ax.imshow(population)
        #   # ax.set_yticks(np.arange(len(clustersNames)))
        #   # ax.set_xticks(np.arange(len(classesIdx)))
        #   # # for y in range(n):
        #   # #   for x in range(10):
        #   # #     label = population[x, y]
        #   # #     text_x = x
        #   # #     text_y = y
        #   # #     ax.text(text_x, text_y, label, color='black', ha='center', va='center')
        #   # cb = plt.colorbar(im)
        #   # # plt.show()
        #   # plt.savefig(
        #   #   'heatmaps_pascal/pureness/popHeatmap_{}_clusters_{}_eigvecs_temp.png'.format(n, vLen))
        #   # plt.close(fig)
        #   # ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        #   # fig, axs = plt.subplots(n, 13)
        #   # fig, axs = plt.subplots(n, 14, figsize=(20, n),
        #   #                         gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 2.5, 5]})
        #   # plt.subplots_adjust(hspace=0.4, wspace=0.2)
        #   # fig.suptitle('{} clusters, {} eigV - MI = {}'.format(n, vLen, MI1), fontsize=30)
        #   # plt.subplots_adjust(wspace=0.1, hspace=0.1, left=2, right=2.2)
        #   sortbyH = [sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0]) for i in range(n)]
        #   sortbyclass = [np.argmax(population[i, :]) for i in range(n)]
        #   sortedclustersIdx = np.lexsort((sortbyH, sortbyclass))
        #   # create similarity hists
        #   patchesBycluster = np.hstack(patches2Include)
        #   # distsFromClustersCenter = clusters.transform(Y)
        #   # distsperCluster = [distsFromClustersCenter[np.where(clusters.labels_ == i)] for i in range(n)]
        #   dis4Hist = np.zeros(dissim.shape)
        #   clustersSize = np.unique([clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)], return_counts=True)
        #   for i in range(len(patchesBycluster)):
        #     dis4Hist[i, :] = affinity[patchesBycluster[i], :]
        #   dissim_forcopy = copy.deepcopy(dis4Hist)
        #   for i in range(len(patchesBycluster)):
        #     dis4Hist[:, i] = dissim_forcopy[:, patchesBycluster[i]]
        #   dis4Hist = dis4Hist[:np.hstack(patches2Include).shape[0], :np.hstack(patches2Include).shape[0]]
        #   dists_within_cluster = []
        #   for i in range(n):
        #     if i == 0:
        #       dists_within_cluster.append(dis4Hist[:clustersSize[1][i], :clustersSize[1][i]])
        #     elif i == n-1:
        #       dists_within_cluster.append(dis4Hist[-clustersSize[1][i]:, -clustersSize[1][i]:])
        #     else:
        #       dists_within_cluster.append(dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                 sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)])])
        #
        #   dists_between_cluster = []
        #   for i in range(n):
        #     if i == 0:
        #       dists_between_cluster.append(dis4Hist[:clustersSize[1][i], clustersSize[1][i]:])
        #     elif i == n-1:
        #       dists_between_cluster.append(dis4Hist[-clustersSize[1][i]:, :-clustersSize[1][i]])
        #     else:
        #       dists_between_cluster.append(np.hstack((dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                             :sum(clustersSize[1][:i])],
        #                                             dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                             sum(clustersSize[1][:(i + 1)]):])))
        #   condH = []
        #   for k in range(n):
        #     i = sortedclustersIdx[k]
        #     idxlist = np.where(clusters.labels_ == i)
        #     #       print(len(idxlist[0])/len(patches))
        #     GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
        #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
        #     # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
        #     clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
        #     condH.append(clusterscondH)
        #   #   barvals = []
        #   #   for c in classesPopDS[0]:
        #   #     if c in classesPop[0]:
        #   #       barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
        #   #     else:
        #   #       barvals.append(0)
        #   #   ticks = range(18)
        #   #   # gs = axs[i, -2].get_gridspec()
        #   #   # remove the underlying axes
        #   #   # for ax in axs[i, 10:]:
        #   #   #   ax.remove()
        #   #   # axbig = fig.add_subplot(gs[i, 10:])
        #   #   axs[k, 0].set_ylabel(
        #   #     'cluster no.{} \n H={:.4f},\n dominant class:{} \n p_class={:.4f}'.format(i, clusterscondH, np.argmax(population[i, :]), np.max(population[i, :])),
        #   #     rotation=0, size='medium', loc='bottom')
        #   #   for j in range(14):
        #   #     if j == 0:
        #   #       plt.setp(axs[k, j].spines.values(), color='white')
        #   #       axs[k, j].set_xticks([])
        #   #       axs[k, j].set_yticks([])
        #   #     elif j < 6:
        #   #       axs[k, j].imshow(ims_true[i][j - 1])
        #   #       axs[k, j].set_xticks([])
        #   #       axs[k, j].set_yticks([])
        #   #       # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
        #   #     elif j == 6:
        #   #       axs[k, j].remove()
        #   #     elif j < 12:
        #   #       try:
        #   #         axs[k, j].imshow(ims_false[i][j - 7])
        #   #         axs[k, j].set_xticks([])
        #   #         axs[k, j].set_yticks([])
        #   #         # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
        #   #       except:
        #   #         plt.setp(axs[k, j].spines.values(), color='white')
        #   #         axs[k, j].set_xticks([])
        #   #         axs[k, j].set_yticks([])
        #   #     elif j==12:
        #   #       # ticks = range(len(classesPop[1]))
        #   #       axs[k, j].bar(ticks, barvals, align='center')
        #   #       axs[k, j].set_xticks(ticks)
        #   #       axs[k, j].tick_params(axis='x', rotation=0)
        #   #       # axs[i,j].hist(GT_cluster, )
        #   #     else:
        #   #       axs[k, j].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
        #   #       axs[k, j].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)
        #   #
        #   # fig.align_ylabels()
        #   # # fig.tight_layout()
        #   # plt.savefig(
        #   #   'Thesis figures/pascal/clustering/noBG/{}_clusters_{}_eigvecs_{:.4f}_MI.png'.format(n, vLen, MI1),
        #   #   bbox_inches='tight')
        #   # plt.close(fig)
        #
        #   conHs[str(n) + ' ' + str(n)] = condH
        #   condHperClass = {}
        #   for m in range(len(classesPopDS[0])):
        #     Hs = []
        #     for i, c in enumerate(patches2Include):
        #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i,:])==np.flatnonzero(list(classesIdx.values())==p['seg']))]
        #       # if np.max(population[i,:]) >= purity:
        #       #   chance2classify[np.argmax(population[i,:])] += len(ps)/classesPopDS[1][np.argmax(population[i,:])]
        #       for p in ps:
        #         # print(type(classesIdx[m]))
        #         if p['seg'] == classesIdx[m]:
        #           Hs.append(condH[i])
        #     condHperClass[m] = Hs
        #   for pu in purity:
        #     chance2classify = np.zeros(len(classesPopDS[0]))
        #     for i, c in enumerate(patches2Include):
        #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i, :]) == np.flatnonzero(
        #         list(classesIdx.values()) == p['seg']))]
        #       if np.max(population[i, :]) >= pu:
        #         chance2classify[np.argmax(population[i, :])] += len(ps) / classesPopDS[1][np.argmax(population[i, :])]
        #     chance2cl[str(n)+'_'+str(n)+'_'+str(pu)] = chance2classify
        #   condHperC[str(n)+'_'+str(n)] = condHperClass
          for vLen in vLens:
            purity_train = []
            purity_test = []
            train_label = []
            test_label = []
            vLen=n
            ims_true = []
            ims_false = []
            space = eigenVectors[:, :vLen]
            normalize_f = cp.sqrt((np.sum(cp.square(space), axis=1)))
            Y = (space.T / normalize_f).T
            Kmeans = sklearn.cluster.KMeans(n_clusters=n, n_init=50)
            im_ids = random.sample(range(np.max([p['im_id'] for p in patches])),
                                   int(np.max([p['im_id'] for p in patches]) * 2/3))
            patches_fit = [p['idx'] for p in patches if p['im_id'] in im_ids]
            fit_sapce = Y[patches_fit, :]
            clusters = Kmeans.fit(fit_sapce)
            patches_predict = [p['idx'] for p in patches if p['im_id'] not in im_ids]
            distsFromClustersCenter = clusters.transform(Y)
            predict_space = Y[patches_predict, :]
            predictions = Kmeans.predict(predict_space)
            # temp = list(clusters.labels_)
            # for i, pred in zip(patches_predict, predictions):
            #   temp.insert(i, pred)
            # clusters.labels_ = np.array(temp)
            distsperCluster = [
              (distsFromClustersCenter[np.array(patches_fit)[np.where(clusters.labels_ == i)]], np.array(patches_fit)[np.where(clusters.labels_ == i)]) for i in
              range(n)]
            patches2Include = [distsperCluster[i][1][np.where(
              distsperCluster[i][0][:, i] <= np.percentile(distsperCluster[i][0][:, i], 100))[0]] for i in range(n)]
            # clusters_trans = Kmeans.fit_transform(Y)
            # clustersCenterD.append(np.mean(
            #   [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
            #    for j in range(n) if i != j]))
            # inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
          # # plot the change in the distances according to the size of the new patches descriptors
          #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
          #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
          #   ax2.set_xlabel('# eig vec')
          #   ax2.set_ylabel('mean distance')
          #   ax2.legend()
            # simulate and test the population of one cluster
            population = np.zeros((n, len(classesPopDS[0])))
            probs = np.zeros(population.shape)
            # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
            # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
            clustersNames = range(n)
            # labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
            #                'scissors background', 'airplane', 'airplane background']
            # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
            #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
            counter = np.zeros((1, n))
            for clusterIDX in range(n):
              # idxlist = np.where(clusters.labels_ == clusterIDX)
              idxlist = np.concatenate([patches2Include[clusterIDX], np.array(patches_predict)[np.where(predictions==clusterIDX)]])
              #       print(len(idxlist[0])/len(patches))
              gt_train = [(p['seg']) for p in patches if p['idx'] in patches2Include[clusterIDX]]
              gt_test = [(p['seg']) for p in patches if
                         p['idx'] in np.array(patches_predict)[np.where(predictions == clusterIDX)]]
              train_pop = np.unique(gt_train, return_counts=True, axis=0)
              test_pop = np.unique(gt_test, return_counts=True, axis=0)
              purity_train.append(np.max(train_pop[1])/sum(train_pop[1]))
              train_label.append(train_pop[0][np.argmax(train_pop[1])])
              try:
                purity_test.append(np.max(test_pop[1])/sum(test_pop[1]))
                test_label.append(test_pop[0][np.argmax(test_pop[1])])
              except:
                purity_test.append(None)
                test_label.append(None)
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist]
              classesPop = np.unique(gt_train, return_counts=True, axis=0) #np.unique(GT_cluster, return_counts=True, axis=0)
              max_Pclass = copy.deepcopy(classesPop[0])[np.argmax(classesPop[1])]
              for i, pop in enumerate(classesPop[0]):
                val = list(classesIdx.values()).index(pop)
                # population[clusterIDX, val] = classesPop[1][i] / len(idxlist[0])
                # if b'bus' in pop and b'255' in pop:
                #   val = 0
                # if b'bus' in pop and b'0' in pop:
                #   val = 1
                # if b'dog' in pop and b'255' in pop:
                #   val = 2
                # if b'dog' in pop and b'0' in pop:
                #   val = 3
                # if b'pizza' in pop and b'255' in pop:
                #   val = 4
                # if b'pizza' in pop and b'0' in pop:
                #   val = 5
                # if b'scissors' in pop and b'255' in pop:
                #   val = 6
                # if b'scissors' in pop and b'0' in pop:
                #   val = 7
                # if b'airplane' in pop and b'255' in pop:
                #   val = 8
                # if b'airplane' in pop and b'0' in pop:
                #   val = 9
                classesPop[0][i] = val
                population[clusterIDX, int(classesPop[0][i])] = classesPop[1][i] / sum(classesPop[1])
                probs[clusterIDX, int(classesPop[0][i])] = classesPop[1][i]/len(patches)
              # max_Pclass = classesPop[0][np.argmax(classesPop[1])]
              try:
                true_im = [p['ref'] for p in patches if
                           (p['idx'] in idxlist and max_Pclass == p['seg'])]
                ims_true.append(random.sample(true_im, k=np.min([5, len(true_im)])))
              except:
                print(max_Pclass)
              false_im = [p['ref'] for p in patches if (
                        p['idx'] in idxlist and max_Pclass != p['seg'])]
              try:
                ims_false.append(random.sample(false_im, k=5))
              except:
                ims_false.append(false_im)
              counter[0, clusterIDX] = np.sum(classesPop[1])
            # #       for i, pop in enumerate(classesPop[0]):
            # #         if b'elephant' in pop and b'255' in pop:
            # #           val = 4
            # #         if b'elephant' in pop and b'0' in pop:
            # #           val = 5
            # #         if b'dog' in pop and b'255' in pop:
            # #           val = 2
            # #         if b'dog' in pop and b'0' in pop:
            # #           val = 3
            # #         if b'fire hydrant' in pop and b'255' in pop:
            # #           val = 6
            # #         if b'fire hydrant' in pop and b'0' in pop:
            # #           val = 7
            # #         if b'train' in pop and b'255' in pop:
            # #           val = 8
            # #         if b'train' in pop and b'0' in pop:
            # #           val = 9
            # #         if b'airplane' in pop and b'255' in pop:
            # #           val = 0
            # #         if b'airplane' in pop and b'0' in pop:
            # #           val = 1
            # #         classesPop[0][i] = val
            # #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
            # #         counter[val] += 1
            MI1 = sklearn.metrics.mutual_info_score([p['seg'] for p in patches if p['idx'] in np.hstack(patches2Include)], clusters.labels_)#[clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)])
            MI_n.append(MI1)
            clusters2show = {}

            sortbyH = [sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0]) for i in range(n)]
            sortbyclass = [np.argmax(population[i, :]) for i in range(n)]
            sortedclustersIdx = np.lexsort((sortbyH, sortbyclass))
            # create similarity hists
            patchesBycluster = np.hstack(patches2Include)
            # distsFromClustersCenter = clusters.transform(Y)
            # distsperCluster = [distsFromClustersCenter[np.where(clusters.labels_ == i)] for i in range(n)]
            for k in range(n):
              i = sortedclustersIdx[k]
              if str(np.argmax(population[i, :])) in clusters2show.keys():
                continue
                # if clusters2show[str(np.argmax(population[i, :]))]==2:
                #   continue
                # else:
                #   clusters2show[str(np.argmax(population[i, :]))] = 2
              else:
                clusters2show[str(np.argmax(population[i, :]))]=1
            numOfRows = sum(clusters2show.values())+5
            fig, axs = plt.subplots(numOfRows, 17, figsize=(numOfRows, 17),
                                    gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 0.2, 2, 0.2, 2]})
            plt.subplots_adjust(hspace=0.4)
            # fig.suptitle('{} clusters, MI = {}'.format(n, MI1))
            # plt.subplots_adjust(wspace=0.05, hspace=0.6)  # , left=2, right=2.2)

            dis4Hist = np.zeros(dissim.shape)
            clustersSize = np.unique([clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)], return_counts=True)
            for i in range(len(patchesBycluster)):
              dis4Hist[i, :] = dissim[patchesBycluster[i], :]
            dissim_forcopy = copy.deepcopy(dis4Hist)
            for i in range(len(patchesBycluster)):
              dis4Hist[:, i] = dissim_forcopy[:, patchesBycluster[i]]
            dis4Hist = dis4Hist[:np.hstack(patches2Include).shape[0], :np.hstack(patches2Include).shape[0]]
            dists_within_cluster = []
            for i in range(n):
              if i == 0:
                dists_within_cluster.append(dis4Hist[:clustersSize[1][i], :clustersSize[1][i]])
              # elif i == n-1:
              #   dists_within_cluster.append(dis4Hist[-clustersSize[1][i]:, -clustersSize[1][i]:])
              else:
                dists_within_cluster.append(dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                          sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)])])

            dists_between_cluster = []
            for i in range(n):
              if i == 0:
                dists_between_cluster.append(dis4Hist[:clustersSize[1][i], clustersSize[1][i]:])
              # elif i == n-1:
              #   dists_between_cluster.append(dis4Hist[-clustersSize[1][i]:, :-clustersSize[1][i]])
              else:
                dists_between_cluster.append(np.hstack((dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                                      :sum(clustersSize[1][:i])],
                                                      dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                                      sum(clustersSize[1][:(i + 1)]):])))
            clusters2show = {}
            condH = []
          ### old plots start here
          #   for k in range(n):
          #     i = sortedclustersIdx[k]
          #     idxlist = np.where(clusters.labels_ == i)
          #     #       print(len(idxlist[0])/len(patches))
          #     GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
          #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
          #     # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
          #     clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
          #     condH.append(clusterscondH)
          #   #   barvals = []
          #   #   for c in classesPopDS[0]:
          #   #     if c in classesPop[0]:
          #   #       barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
          #   #     else:
          #   #       barvals.append(0)
          #   #   ticks = range(18)
          #   #   # gs = axs[i, -2].get_gridspec()
          #   #   # remove the underlying axes
          #   #   # for ax in axs[i, 10:]:
          #   #   #   ax.remove()
          #   #   # axbig = fig.add_subplot(gs[i, 10:])
          #   #   axs[k, 0].set_ylabel(
          #   #     'cluster no.{} \n H={:.4f},\n dominant class:{} \n p_class={:.4f}'.format(i, clusterscondH, np.argmax(population[i, :]), np.max(population[i, :])),
          #   #     rotation=0, size='medium', loc='bottom')
          #   #   for j in range(14):
          #   #     if j == 0:
          #   #       plt.setp(axs[k, j].spines.values(), color='white')
          #   #       axs[k, j].set_xticks([])
          #   #       axs[k, j].set_yticks([])
          #   #     elif j < 6:
          #   #       axs[k, j].imshow(ims_true[i][j - 1])
          #   #       axs[k, j].set_xticks([])
          #   #       axs[k, j].set_yticks([])
          #   #       # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
          #   #     elif j == 6:
          #   #       axs[k, j].remove()
          #   #     elif j < 12:
          #   #       try:
          #   #         axs[k, j].imshow(ims_false[i][j - 7])
          #   #         axs[k, j].set_xticks([])
          #   #         axs[k, j].set_yticks([])
          #   #         # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
          #   #       except:
          #   #         plt.setp(axs[k, j].spines.values(), color='white')
          #   #         axs[k, j].set_xticks([])
          #   #         axs[k, j].set_yticks([])
          #   #     elif j==12:
          #   #       # ticks = range(len(classesPop[1]))
          #   #       axs[k, j].bar(ticks, barvals, align='center')
          #   #       axs[k, j].set_xticks(ticks)
          #   #       axs[k, j].tick_params(axis='x', rotation=0)
          #   #       # axs[i,j].hist(GT_cluster, )
          #   #     else:
          #   #       axs[k, j].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
          #   #       axs[k, j].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)
          #   #
          #   # fig.align_ylabels()
          #   # # fig.tight_layout()
          #   # plt.savefig(
          #   #   'Thesis figures/pascal/clustering/noBG/{}_clusters_{}_eigvecs_{:.4f}_MI.png'.format(n, vLen, MI1),
          #   #   bbox_inches='tight')
          #   # plt.close(fig)
          #
          #   conHs[str(n) + ' ' + str(vLen)] = condH
          #   condHperClass = {}
          #   for m in range(len(classesPopDS[0])):
          #     Hs = []
          #     for i, c in enumerate(patches2Include):
          #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i,:])==np.flatnonzero(list(classesIdx.values())==p['seg']))]
          #       # if np.max(population[i,:]) >= purity:
          #       #   chance2classify[np.argmax(population[i,:])] += len(ps)/classesPopDS[1][np.argmax(population[i,:])]
          #       for p in ps:
          #         # print(type(classesIdx[m]))
          #         if p['seg'] == classesIdx[m]:
          #           Hs.append(condH[i])
          #     condHperClass[m] = Hs
          #   for pu in purity:
          #     chance2classify = np.zeros(len(classesPopDS[0]))
          #     cov = np.zeros(len(classesPopDS[0]))
          #     for i, c in enumerate(patches2Include):
          #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i, :]) == np.flatnonzero(
          #         list(classesIdx.values()) == p['seg']))]
          #       if np.max(population[i, :]) >= pu:
          #         chance2classify[np.argmax(population[i, :])] += len(ps) / classesPopDS[1][np.argmax(population[i, :])]
          #         cov[np.argmax(population[i, :])] += len(ps)
          #     chance2cl[str(n)+'_'+str(pu)] = chance2classify
          #     coverage[str(vLen)+'_'+str(pu)] = cov
          #   condHperC[str(n)+'_'+str(vLen)] = condHperClass
          # MIs[str(n)] = MI_n
            for k in range(n):
              ind = 0
              if clusters2show.values():
                ind = sum(clusters2show.values())
              i = sortedclustersIdx[k]

              idxlist = np.concatenate([np.array(patches_fit)[np.where(clusters.labels_ == i)], np.array(patches_predict)[np.where(predictions==i)]])
              #       print(len(idxlist[0])/len(patches))
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist]
              gt_train = [(p['seg']) for p in patches if p['idx'] in np.array(patches_fit)[np.where(clusters.labels_ == i)]]
              gt_test = [(p['seg']) for p in patches if
                         p['idx'] in np.array(patches_predict)[np.where(predictions == i)]]
              classesPop = np.unique(gt_train, return_counts=True, axis=0)
              # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
              clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
              condH.append(clusterscondH)
              if str(np.argmax(population[i, :])) in clusters2show.keys():
                continue
                # if clusters2show[str(np.argmax(population[i, :]))]==2:
                #   continue
                # else:
                #   clusters2show[str(np.argmax(population[i, :]))] = 2
              else:
                clusters2show[str(np.argmax(population[i, :]))]=1
              barvals = []
              for c in classesPopDS[0]:
                if c in classesPop[0]:
                  barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
                else:
                  barvals.append(0)
            #   ticks = range(18)
            #   # gs = axs[i, -2].get_gridspec()
            #   # remove the underlying axes
            #   # for ax in axs[i, 10:]:
            #   #   ax.remove()
            #   # axbig = fig.add_subplot(gs[i, 10:])
              axs[ind, 0].set_ylabel(
                'cluster no.{} \n H={:.4f} #im={}'.format(i, clusterscondH, len(np.unique([(p['im_id']) for p in patches if p['idx'] in idxlist], return_counts=True)[1])),
                rotation=0, size='medium', loc='center')
              axs[ind, 1].set_ylabel(
                'dominant class:{} \n p_class={:.4f}'.format(classesPopDS[0][np.argmax(population[i, :])], np.max(population[i, :])),
                rotation=0, size='medium', loc='center')
              for j in range(15):
                if j == 0 or j==1:
                  plt.setp(axs[ind, j].spines.values(), color='white')
                  axs[ind, j].set_xticks([])
                  axs[ind, j].set_yticks([])
                elif j < 7:
                  try:
                    axs[ind, j].imshow(ims_true[i][j - 2])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                  # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
                elif j == 7:
                  axs[ind, j].remove()
                elif j < 13:
                  try:
                    axs[ind, j].imshow(ims_false[i][j - 8])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                    # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                elif j==13:
                  axs[ind, j].remove()
                elif j < 15:
                  ticks = range(len(barvals))
                  axs[ind, j].bar(ticks, barvals, align='center')
                  # axs[ind, j].set_xticks(ticks)
                  axs[ind, j].tick_params(axis='x', rotation=0)
                  # axs[i,j].hist(GT_cluster, )
                  axs[ind, j+1].remove()
                  axs[ind, j+2].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
                  axs[ind, j+2].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)

            for ind, k in enumerate(np.argsort(sortbyH)[-5:]):

              i = k
              ind += sum(clusters2show.values())
              idxlist = np.where(clusters.labels_ == i)[0]
              #       print(len(idxlist[0])/len(patches))
              print(idxlist)
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist]
              classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
              # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
              clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
              condH.append(clusterscondH)
              barvals = []
              for c in classesPopDS[0]:
                if c in classesPop[0]:
                  barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
                else:
                  barvals.append(0)
            #   ticks = range(18)
            #   # gs = axs[i, -2].get_gridspec()
            #   # remove the underlying axes
            #   # for ax in axs[i, 10:]:
            #   #   ax.remove()
            #   # axbig = fig.add_subplot(gs[i, 10:])
              axs[ind, 0].set_ylabel(
                'cluster no.{} \n H={:.4f} #im={}' .format(i, clusterscondH, len(np.unique([(p['im_id']) for p in patches if p['idx'] in idxlist], return_counts=True)[1])),
                rotation=0, size='small', loc='center')
              axs[ind, 1].set_ylabel(
                'dominant class:{} \n p_class={:.4f}'.format(classesPopDS[0][np.argmax(population[i, :])],
                                                                                          np.max(population[i, :])),
                rotation=0, size='small', loc='center')
              for j in range(15):
                if j == 0 or j==1:
                  plt.setp(axs[ind, j].spines.values(), color='white')
                  axs[ind, j].set_xticks([])
                  axs[ind, j].set_yticks([])
                elif j < 7:
                  try:
                    axs[ind, j].imshow(ims_true[i][j - 2])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                  # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
                elif j == 7:
                  axs[ind, j].remove()
                elif j < 13:
                  try:
                    axs[ind, j].imshow(ims_false[i][j - 8])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                    # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                elif j == 13:
                  axs[ind, j].remove()
                elif j<15:
                  ticks = range(len(barvals))
                  axs[ind, j].bar(ticks, barvals, align='center')
                  # axs[ind, j].set_xticks(ticks)
                  axs[ind, j].tick_params(axis='x', rotation=0)
                  # axs[i,j].hist(GT_cluster, )
                  axs[ind, j+1].remove()
                  axs[ind, j+2].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
                  axs[ind, j+2].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)
            fig.align_ylabels()
            # fig.tight_layout()
            plt.savefig(
              'Thesis figures/pascal/samples/filtered_{}_clusters_{:.4f}_MI.png'.format(n, MI1),
              bbox_inches='tight')
            # plt.show()
            plt.close(fig)

            conHs[str(n) + ' ' + str(vLen)] = condH
            condHperClass = {}
            for m in range(len(classesPopDS[0])):
              Hs = []
              for i, c in enumerate(patches2Include):
                ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i,:])==list(classesIdx.values()).index(p['seg']))]
                # if np.max(population[i,:]) >= purity:
                #   chance2classify[np.argmax(population[i,:])] += len(ps)/classesPopDS[1][np.argmax(population[i,:])]
                for p in ps:
                  # print(type(classesIdx[m]))
                  if p['seg'] == classesIdx[m]:
                    Hs.append(condH[i])
              condHperClass[m] = Hs
            for pu in purity:
              chance2classify = np.zeros(len(classesPopDS[0]))
              cov = np.zeros(len(classesPopDS[0]))
              for i, c in enumerate(patches2Include):
                ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i, :]) == list(classesIdx.values()).index(p['seg']))]
                if np.max(population[i, :]) >= pu:
                  chance2classify[np.argmax(population[i, :])] += len(ps) / classesPopDS[1][np.argmax(population[i, :])]
                  cov[np.argmax(population[i, :])] += len(ps)
              chance2cl[str(n)+'_'+str(pu)] = chance2classify
              coverage[str(vLen)+'_'+str(pu)] = cov
            condHperC[str(n)+'_'+str(vLen)] = condHperClass
          MIs[str(n)] = MI_n



            # plt.show()
          # # simulate and test the population of one cluster
          #   population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
          #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
          #   clustersNames = range(n)
          #   labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background','elephant',
          #   'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
          #   for clusterIDX in range(n):
          #     idxlist = np.where(clusters.labels_ == clusterIDX)
          #     print(len(idxlist)/len(patches))
          #     GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
          #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
          #     for i, pop in enumerate(classesPop[0]):
          #       if b'airplane' in pop and b'255' in pop:
          #         classesPop[0][i] = 0
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'airplane' in pop and b'0' in pop:
          #         classesPop[0][i] = 1
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'dog' in pop and b'255' in pop:
          #         classesPop[0][i] = 2
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'dog' in pop and b'0' in pop:
          #         classesPop[0][i] = 3
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'fire hydrant' in pop and b'255' in pop:
          #         classesPop[0][i] = 6
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'fire hydrant' in pop and b'0' in pop:
          #         classesPop[0][i] = 7
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'elephant' in pop and b'255' in pop:
          #         classesPop[0][i] = 4
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'elephant' in pop and b'0' in pop:
          #         classesPop[0][i] = 5
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'train' in pop and b'255' in pop:
          #         classesPop[0][i] = 8
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'train' in pop and b'0' in pop:
          #         classesPop[0][i] = 9
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #   fig, ax = plt.subplots()
          #   im = ax.imshow(population)
          #   ax.set_yticks(np.arange(len(clustersNames)))
          #   ax.set_xticks(np.arange(len(labelsNames)))
          #   cb = plt.colorbar(im)
          #   plt.savefig('heatmaps_coco_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
          #   plt.close(fig)
            # population[clusterIDX, classesPop[0]] = classesPop[1]
            # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
            #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])

          # plt.show()
          # plt.savefig('keepDaway_newpatches.png')
          # plt.close(fig2)
          # fig, ax = plt.subplots()
          # ax.imshow(population)
          # ax.set_yticks(np.arange(len(clustersNames)), labels=clustersNames)
          # ax.set_xticks(np.arange(len(labelsNames)), labels=labelsNames)
        ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
        random.shuffle(ims)
        fig, axs = plt.subplots(6, 6)
        for i in range(6):
          for j in range(6):
            axs[i, j].imshow(ims[i * 6 + j])
        plt.show()
          # patches = sorted(patches, key=lambda p: p['idx'])
          # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
          # clustering_kmeans = SpectralClustering(n_clusters=50,
          #                                        assign_labels='kmeans',
          #                                        random_state=0,
          #                                        affinity='nearest_neighbors').fit(
          #   np.array([p['patch'].flatten() for p in patches]))
          # clustering_disc = SpectralClustering(n_clusters=50,
          #                                      assign_labels='discretize',
          #                                      random_state=0,
          #                                      affinity='nearest_neighbors').fit(
          #   np.array([p['patch'].flatten() for p in patches]))
          #
          # plt.hist(clustering_disc.labels_, bins=50)
          # plt.hist(clustering_kmeans.labels_, bins=50)
  elif pascal_sparse:
    # image, mask, imname = getRep_pascal()

    # _, _, fuse = resnet50.inference(image, tf.zeros(
    #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    #    resnet50_input.NUM_CLASSES]), tf.zeros(
    #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    #    resnet50_input.NUM_CLASSES]))
    # saver = tf.train.Saver(tf.global_variables())
    # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
    #                                       intra_op_parallelism_threads=1)) as sess:
    #   sess.run(tf.group(tf.initialize_all_variables(),
    #                     tf.initialize_local_variables()))
    #   ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    #   if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #   else:
    #     print('No checkpoint file found')
    #     return
    #   Start the queue runners.
      # coord = tf.train.Coordinator()
    # #
    # # _, _, fuse = resnet50.inference(image, tf.zeros(
    # #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    # #    resnet50_input.NUM_CLASSES]), tf.zeros(
    # #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    # #    resnet50_input.NUM_CLASSES]))
    # # # ### bsds model
    # # saver = tf.train.Saver(tf.global_variables())
    # # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
    # #                                       intra_op_parallelism_threads=1)) as sess:
    # #   sess.run(tf.group(tf.initialize_all_variables(),
    # #                     tf.initialize_local_variables()))
    # #   ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # #   if ckpt and ckpt.model_checkpoint_path:
    # #     saver.restore(sess, ckpt.model_checkpoint_path)
    # #   else:
    # #     print('No checkpoint file found')
    # #     return
    # # # ### pascal weights
    # # # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
    # # #                                       intra_op_parallelism_threads=1)) as sess:
    # # #   model_dict = np.load('fp_weights.npy', allow_pickle=True).item()
    # # #   all_vars = tf.trainable_variables()
    # # #   for v in all_vars:
    # # #     if (v.op.name.find("weights") > -1) and (v.op.name.find("softmax_linear") == -1) and (
    # # #             v.op.name.find("fuse") == -1) and (v.op.name.find("conv5_3") == -1):
    # # #       assign_op = v.assign(model_dict[v.op.name])
    # # #       sess.run(assign_op)
    # # #
    # #   sess.run(tf.group(tf.initialize_all_variables(),
    # #                     tf.initialize_local_variables()))
    # # #   # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # # #   # if ckpt and ckpt.model_checkpoint_path:
    # # #   #   saver.restore(sess, ckpt.model_checkpoint_path)
    # # #   # else:
    # # #   #   print('No checkpoint file found')
    # # #   #   return
    # # #   # Start the queue runners.
    # #   coord = tf.train.Coordinator()
    #   try:
    #     threads = []
    #     for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #       threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
    #                                        start=True))
    #     patches = getPatchDS_pascal(sess, image, mask, imname, fuse, patch_size=8, ifPCA=True, n=16, BG=False)
        # patches = getPatchDS_frompkl()
        #
        # with open('patches_8_pascal_pca16_balanced_noBG_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(patches, f)

        with open('patches_8_pascal_pca16_balanced_noBG_2025-02-02 12:34:41.083454.pkl', 'rb') as f:
          patches = pickle.load(f)
        # with open('patches_8_pascal_pca16_balanced_noBG_2024-12-15 13:06:11.715102.pkl', 'rb') as f:
        #   patches = pickle.load(f)
        # filtered_patches = []
        # remain_patches = patches[:10000]
        # while remain_patches:
        #   p = remain_patches[0]
        #   centroids, ids = getCentroid(p, remain_patches)
        #   for c in centroids:
        #     c['temp_idx']=len(filtered_patches)
        #     filtered_patches.append(c)
        #   remain_patches = remain_patches[len(ids):]
        # dissim = compute_all_dissim_parallel(patches, block_size=5000, percentile=10, max_workers=8)
        # N = len(patches)
        # Open the same raw file in read-only or read-write mode:
        with open("csr_dissim.pkl", "rb") as f:
          dissim = pickle.load(f)
        # dissim = create_dissim_sparse(patches)
        print(dissim.shape)
        # scores = dist_methods_comparison(patches)
        # with open('patches_8_pascal_pca16_balanced_noBG_2024-03-07 12:22:39.858088.pkl', 'rb') as f:
        #   patches = pickle.load(f)
        # classes2inc = [82, 99, 123]
        # patches = [p for p in patches if p['seg'] in classes2inc]
        # for i in range(len(patches)):
        #   patches[i]['idx']=i
        # new_patches = random.sample(patches, 1800)
        # with open('patches_8_pascal_pca16_filtered_sampled.pkl', 'rb') as f:
        #   new_patches = pickle.load(f)
        # lda = LDA(new_patches)
  #     # dissim =
  # #
  # # # # #       # patches = random.choices(patches, k=500)
  # # # # #       # threads_per_block = (16,16)
  # # # # #       # blocks_per_grid = ((4224+16-1)//16, (4224+16-1)//16)
  # # # # #       # patches_array = [p['patch'] for p in patches]
  # # # # #       # patches_cuda = cuda.to_device(patches_array)
  #       doues = comb(new_patches, 2)
  # #       # doues = [(d, lda) for d in doues]
  # # # # # # # # #       # dissim = calcPatchDistmulti(doues)
  #       data = []
  #       labels = []
  # #       # p1order = []
  # #       # p2order = []
  #       with multiprocessing.Pool(processes=16) as p:
  #           with tqdm(total=(len(new_patches) ** 2)/2) as pbar:
  #             for out in p.imap_unordered(calcPatchDist4lda, doues):
  #               pbar.update()
  #               _, _, d, l = out
  #               # p1 = np.where([p['idx']==p1 for p in new_patches])
  #               # p2 = np.where([p['idx']==p2 for p in new_patches])
  #               # p1order.append(p1)
  #               # p2order.append(p2)
  #               data.append(d)
  #               labels.append(int(l))
  #       data = np.array(data)
  #       labels = np.array(labels)
  #       weights = lda_fromScratch(data, labels)
  #       patches = [p for p in patches if p['seg']!=np.uint8(30)]
  #       with open('Thesis figures/dissim_8_pascal_noBG_q10_fullDS.pkl', 'rb') as f:
          # with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
          #   dissim = pickle.load(mmapped_file)
          # dissim = pickle.load(f)
        # dissim = pd.read_pickle('Thesis figures/dissim_8_pascal_noBG_q10_fullDS.pkl')
        # patches = patches[:1000]
        # doues = itertools.product(patches[11000:15000], repeat=2)
        # dissim = np.zeros((len(patches), len(patches)))

        # for i in tqdm(range(len(patches))):
        #   combinations = list(itertools.product([patches[i]], patches))
        # combinations_dict = {first: [(first, second) for (first, second) in combinations if first == pair[0]] for pair in
        #                combinations}

        # dissim = np.load('dissim_8_pascal_noBG_q10_fullDS.npz')['arr_0']
        # doues = [d for d in doues if dissim[d[0]['idx'], d[1]['idx']] == 0]
        # dists = []
  #       with multiprocessing.Pool(processes=16) as p:
  #         with tqdm(total=(len(patches) ** 2) / 2) as pbar:
  #           for d in p.imap_unordered(calcPatchDist4lda, doues):
  #             pbar.update()
  #             p1, p2, dist, l = d
              # dissim[p1,p2] = dissim[p2,p1] = np.array(dist).dot(weights)
        # with multiprocessing.Pool(processes=16) as p:
        #     with tqdm(total=(len(patches) ** 2)/2) as pbar:
        #       for d in p.imap_unordered(transorm2LDA, doues):
        #         pbar.update()
        #         dissim[d[0], d[1]] = d[2]
        #         dissim[d[1], d[0]] = d[2]
  # # # # #       # dissim = cuda.to_device(dissim)
  # # # # #       # dissim = calcDistGPUPar[blocks_per_grid,threads_per_block](patches_cuda, dissim)
  #       counter = 11000
  #       dists = np.zeros(4000)
  #       with multiprocessing.Pool(processes=16) as p:
  #         with tqdm(total=(len(patches) ** 2)) as pbar:
  #           for d in p.map(calcPatchDistmulti, doues):
  #             pbar.update()
  # #             dists[d[1]-11000] = d[2]
  # #             if d[1]==14999:#len(patches)-1:
  # #               perc = np.percentile(dists, 3)
  # #               idxs = np.where(dists<=perc) #np.logical_and(,dists > 0))
  # #               for idx in idxs[0]:
  # #                 dissim[counter, idx+11000] = dists[idx]
  # #                 dissim[idx+11000, counter] = dists[idx]
  # #               counter += 1
  # #               dists = np.zeros(4000)
  #               # dists = []
  #               dissim[d[0], d[1]] = d[2]
  #               dissim[d[1], d[0]] = d[2]
        # try:

            # if pbar.n % 100000000==0:
            #   np.savez_compressed('dissim_8_pascal_noBG_q10_fullDS.npz', dissim)
            # with open('dissim_8_pascal_noBG_q10_fullDS.pkl', 'wb') as f:
            #   pickle.dump(dissim, f, protocol=4)
        # np.savez_compressed('dissim_8_pascal_noBG_q10_fullDS.npz', dissim)
        # except:
        #   continue
        #       # dists.append([d[3], (d[0], d[1])])
  # # # #       # # for p1 in tqdm(patches):
  # # # #       # #   for p2 in patches:
  # # # #       # #     if dissim[p1['idx'], p2['idx']] != 0:
  # # # #       # #       continue
  # # # #       # #     else:
  # # # #       # #       # start = time.time()
  # # # #       # #       # patch1 = torch.from_numpy(p1['patch'])
  # # # #       # #       # patch2 = torch.from_numpy(p2['patch'])
  # # # #       # #       # dist = calcPatchDisttorch(patch1, patch2)
  # # # #       # #       dist = calcPatchDist(p1['patch'], p2['patch'])
  # # # #       # #       dissim[p1['idx'], p2['idx']] = dist
  # # # #       # #       dissim[p2['idx'], p1['idx']] = dist
  # # # #       # #       # print(time.time()-start)
  #       dissim = dissim[np.any(dissim != 0, axis=1), :][:, np.any(dissim != 0, axis=1)]


        # with open('dissim_8_pascal_noBG_q10_balanced_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(dissim, f)
        # with open('dissim_8_pascal_noBG_q10_balanced_2025-01-20 13:08:39.986150.pkl', 'rb') as f:
        #   dissim = pickle.load(f)
        # with open('dissim_8_minipascal_avg_filtered_q0_noBG_2024-02-07 11:26:37.549013.pkl', 'rb') as f:
        #   dissim = pickle.load(f)
        # with open('dissim_8_pascal_avg_filtered_q0_noBG_balanced_2024-03-07 13:00:44.615576.pkl', 'rb') as f:
        #   dissim = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q50_2023-09-28 14:59:31.762212.pkl', 'rb') as f:
  #         dissim_q50 = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q80_2023-09-27 16:45:36.574020.pkl', 'rb') as f:
  #         dissim_q80 = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q100_2023-09-27 17:38:53.305407.pkl', 'rb') as f:
  #         dissim_q100 = pickle.load(f)
  #       max_d = np.amax(dissim)
  #       # affinity = 1 - (dissim / max_d)
  #       sigma = np.percentile(dissim, 1)
  #       # dissim[dissim>np.percentile(dissim, 10)]=np.amax(dissim)
  #       affinity = np.exp(-dissim/sigma)
  #
  #       # # affinity[affinity<np.percentile(affinity, 95)]=0
  #       # deg = np.array(A.sum(axis=1)).flatten()
  #       # inv_sqrt = 1.0 / np.sqrt(deg)
  #       # inv_sqrt[np.isinf(inv_sqrt)] = 0.0
  #       #
  #       # def matvec(v):
  #       #   # Step 1: u = D^{-1/2} * v
  #       #   u = inv_sqrt * v  # elementwise scale
  #       #
  #       #   # Step 2: w = A.dot(u)
  #       #   w = A.dot(u)  # same as affinity @ u
  #       #
  #       #   # Step 3: result = v - D^{-1/2} * w
  #       #   return v - inv_sqrt * w
  #       #
  #       #   # — (c) Wrap into a LinearOperator
  #       #
  #       # N = affinity.shape[0]
  #       # L_op = LinearOperator((N, N), matvec=matvec, dtype=np.float32)
  #       #
  #       # # — (d) Compute only the k smallest eigenpairs (instead of all N):
  #       # k = 10
  #       # eigvals, eigvecs = eigsh(L_op, k=k, which='SM')
  #
  #       D = np.zeros(affinity.shape)
  #       for i in range(affinity.shape[0]):
  #         D[i, i] = sum(affinity[i, :])
  #       L = D- affinity
  #       D_vec = np.diag(D)  # shape (N,)
  #       D_inv_sqrt = 1.0 / np.sqrt(D_vec + 1e-10)  # regularization for stability
  #
  #       # Compute L_sym efficiently:
  #       L_sym = L * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
  #       L_sym = (np.diag(np.diag(D) ** (-1 / 2)) @ L) @ np.diag(np.diag(D) ** (-1 / 2))
  #       eig = scipy.linalg.eig(L_sym)
  #       idx = eig[0].argsort()
  #       eigenValues = eig[0][idx]
  #       eigenVectors = eig[1][:, idx]
        sigma = np.percentile(dissim.data, 1)  # only use nonzero values for percentile

        # Step 2: Compute affinity matrix A = exp(-dissim / sigma)
        affinity = dissim.copy()
        affinity.data = np.exp(-affinity.data / sigma)  # still CSR

        # Step 3: Compute degree matrix D as sparse diagonal
        degrees = np.array(affinity.sum(axis=1)).flatten()
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees + 1e-10))  # Avoid division by zero

        # Step 4: Compute symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
        L_sym = identity(affinity.shape[0]) - D_inv_sqrt @ affinity @ D_inv_sqrt

        # Step 5: Compute eigenvectors/values of L_sym
        # k = 10  # Number of eigenvectors/clusters

        # with open('eig_balanced_pascal_pca16_filtered_noBG_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(eig, f)
        # vLens = range(10, 151, 10)
        vLens = [20]
        clustersCenterD = []
        inClustersD = []
        ns = range(50,201,50)

        # fig2, ax2 = plt.subplots()
        segvals = [(p['seg']) for p in patches]
        classesPopDS = np.unique(segvals, return_counts=True, axis=0)
        classesIdx = {i: classesPopDS[0][i] for i in range(classesPopDS[0].shape[0])}
        MIs = {}
        conHs = {}
        condHperC = {}
        purity = np.arange(0.1, 1, 0.1)
        chance2cl = {}
        coverage = {}
        for n in tqdm(ns):
          MI_n = []
        #   ims_true = []
        #   ims_false = []
        #   space = eig[1][:, :n]
        #   normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
        #   Y = (space.T / normalize_f).T
        #
        #   Kmeans = sklearn.cluster.KMeans(n_clusters=n, n_init=500)
        #   clusters = Kmeans.fit(Y)
        #   distsFromClustersCenter = clusters.transform(Y)
        #   distsperCluster = [
        #     (distsFromClustersCenter[np.where(clusters.labels_ == i)], np.where(clusters.labels_ == i)) for i in
        #     range(n)]
        #   patches2Include = [distsperCluster[i][1][0][np.where(
        #     distsperCluster[i][0][:, i] <= np.percentile(distsperCluster[i][0][:, i], 100))[0]] for i in range(n)]
        #   # clusters_trans = Kmeans.fit_transform(Y)
        #   # clustersCenterD.append(np.mean(
        #   #   [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
        #   #    for j in range(n) if i != j]))
        #   # inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
        # # # plot the change in the distances according to the size of the new patches descriptors
        # #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
        # #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
        # #   ax2.set_xlabel('# eig vec')
        # #   ax2.set_ylabel('mean distance')
        # #   ax2.legend()
        #   # simulate and test the population of one cluster
        #   population = np.zeros((n, len(classesPopDS[0])))
        #   probs = np.zeros(population.shape)
        #   # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
        #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
        #   clustersNames = range(n)
        #   # labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
        #   #                'scissors background', 'airplane', 'airplane background']
        #   # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
        #   #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
        #   counter = np.zeros((1, n))
        #   for clusterIDX in range(n):
        #     # idxlist = np.where(clusters.labels_ == clusterIDX)
        #     idxlist = [patches2Include[clusterIDX]]
        #     #       print(len(idxlist[0])/len(patches))
        #     GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
        #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
        #     max_Pclass = copy.deepcopy(classesPop[0])[np.argmax(classesPop[1])]
        #     for i, pop in enumerate(classesPop[0]):
        #       val = list(classesIdx.values()).index(pop)
        #       # population[clusterIDX, val] = classesPop[1][i] / len(idxlist[0])
        #       # if b'bus' in pop and b'255' in pop:
        #       #   val = 0
        #       # if b'bus' in pop and b'0' in pop:
        #       #   val = 1
        #       # if b'dog' in pop and b'255' in pop:
        #       #   val = 2
        #       # if b'dog' in pop and b'0' in pop:
        #       #   val = 3
        #       # if b'pizza' in pop and b'255' in pop:
        #       #   val = 4
        #       # if b'pizza' in pop and b'0' in pop:
        #       #   val = 5
        #       # if b'scissors' in pop and b'255' in pop:
        #       #   val = 6
        #       # if b'scissors' in pop and b'0' in pop:
        #       #   val = 7
        #       # if b'airplane' in pop and b'255' in pop:
        #       #   val = 8
        #       # if b'airplane' in pop and b'0' in pop:
        #       #   val = 9
        #       classesPop[0][i] = val
        #       population[clusterIDX, int(classesPop[0][i])] = classesPop[1][i] / sum(classesPop[1])
        #       probs[clusterIDX, int(classesPop[0][i])] = classesPop[1][i]/len(patches)
        #     # max_Pclass = classesPop[0][np.argmax(classesPop[1])]
        #     try:
        #       true_im = [p['ref'] for p in patches if
        #                  (p['idx'] in idxlist[0] and int(max_Pclass) == p['seg'])]
        #       ims_true.append(random.sample(true_im, k=5))
        #     except:
        #       print(max_Pclass)
        #     false_im = [p['ref'] for p in patches if (
        #               p['idx'] in idxlist[0] and int(max_Pclass) != p['seg'])]
        #     try:
        #       ims_false.append(random.sample(false_im, k=5))
        #     except:
        #       ims_false.append(false_im)
        #     counter[0, clusterIDX] = np.sum(classesPop[1])
        #   # #       for i, pop in enumerate(classesPop[0]):
        #   # #         if b'elephant' in pop and b'255' in pop:
        #   # #           val = 4
        #   # #         if b'elephant' in pop and b'0' in pop:
        #   # #           val = 5
        #   # #         if b'dog' in pop and b'255' in pop:
        #   # #           val = 2
        #   # #         if b'dog' in pop and b'0' in pop:
        #   # #           val = 3
        #   # #         if b'fire hydrant' in pop and b'255' in pop:
        #   # #           val = 6
        #   # #         if b'fire hydrant' in pop and b'0' in pop:
        #   # #           val = 7
        #   # #         if b'train' in pop and b'255' in pop:
        #   # #           val = 8
        #   # #         if b'train' in pop and b'0' in pop:
        #   # #           val = 9
        #   # #         if b'airplane' in pop and b'255' in pop:
        #   # #           val = 0
        #   # #         if b'airplane' in pop and b'0' in pop:
        #   # #           val = 1
        #   # #         classesPop[0][i] = val
        #   # #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #   # #         counter[val] += 1
        #   MI1 = sklearn.metrics.mutual_info_score([p['seg'] for p in patches if p['idx'] in np.hstack(patches2Include)], [clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)])
        #   MI_n.append(MI1)
        #   # MI2 = 0
        #   # for i in range(probs.shape[0]):
        #   #   for j in range(probs.shape[1]):
        #   #     if (probs[i, j] * sum(probs[:, j]) * sum(probs[i, :])) != 0:
        #   #       MI2 += probs[i, j] * np.log(probs[i, j] / (sum(probs[:, j]) * sum(probs[i, :])))
        #
        #   # fig, ax = plt.subplots(figsize=(6, 20))
        #   # # counter = np.broadcast_to(counter.T, population.shape)
        #   # # population = np.divide(population, counter)
        #   # im = ax.imshow(population)
        #   # ax.set_yticks(np.arange(len(clustersNames)))
        #   # ax.set_xticks(np.arange(len(classesIdx)))
        #   # # for y in range(n):
        #   # #   for x in range(10):
        #   # #     label = population[x, y]
        #   # #     text_x = x
        #   # #     text_y = y
        #   # #     ax.text(text_x, text_y, label, color='black', ha='center', va='center')
        #   # cb = plt.colorbar(im)
        #   # # plt.show()
        #   # plt.savefig(
        #   #   'heatmaps_pascal/pureness/popHeatmap_{}_clusters_{}_eigvecs_temp.png'.format(n, vLen))
        #   # plt.close(fig)
        #   # ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        #   # fig, axs = plt.subplots(n, 13)
        #   # fig, axs = plt.subplots(n, 14, figsize=(20, n),
        #   #                         gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 2.5, 5]})
        #   # plt.subplots_adjust(hspace=0.4, wspace=0.2)
        #   # fig.suptitle('{} clusters, {} eigV - MI = {}'.format(n, vLen, MI1), fontsize=30)
        #   # plt.subplots_adjust(wspace=0.1, hspace=0.1, left=2, right=2.2)
        #   sortbyH = [sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0]) for i in range(n)]
        #   sortbyclass = [np.argmax(population[i, :]) for i in range(n)]
        #   sortedclustersIdx = np.lexsort((sortbyH, sortbyclass))
        #   # create similarity hists
        #   patchesBycluster = np.hstack(patches2Include)
        #   # distsFromClustersCenter = clusters.transform(Y)
        #   # distsperCluster = [distsFromClustersCenter[np.where(clusters.labels_ == i)] for i in range(n)]
        #   dis4Hist = np.zeros(dissim.shape)
        #   clustersSize = np.unique([clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)], return_counts=True)
        #   for i in range(len(patchesBycluster)):
        #     dis4Hist[i, :] = affinity[patchesBycluster[i], :]
        #   dissim_forcopy = copy.deepcopy(dis4Hist)
        #   for i in range(len(patchesBycluster)):
        #     dis4Hist[:, i] = dissim_forcopy[:, patchesBycluster[i]]
        #   dis4Hist = dis4Hist[:np.hstack(patches2Include).shape[0], :np.hstack(patches2Include).shape[0]]
        #   dists_within_cluster = []
        #   for i in range(n):
        #     if i == 0:
        #       dists_within_cluster.append(dis4Hist[:clustersSize[1][i], :clustersSize[1][i]])
        #     elif i == n-1:
        #       dists_within_cluster.append(dis4Hist[-clustersSize[1][i]:, -clustersSize[1][i]:])
        #     else:
        #       dists_within_cluster.append(dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                 sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)])])
        #
        #   dists_between_cluster = []
        #   for i in range(n):
        #     if i == 0:
        #       dists_between_cluster.append(dis4Hist[:clustersSize[1][i], clustersSize[1][i]:])
        #     elif i == n-1:
        #       dists_between_cluster.append(dis4Hist[-clustersSize[1][i]:, :-clustersSize[1][i]])
        #     else:
        #       dists_between_cluster.append(np.hstack((dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                             :sum(clustersSize[1][:i])],
        #                                             dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                             sum(clustersSize[1][:(i + 1)]):])))
        #   condH = []
        #   for k in range(n):
        #     i = sortedclustersIdx[k]
        #     idxlist = np.where(clusters.labels_ == i)
        #     #       print(len(idxlist[0])/len(patches))
        #     GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
        #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
        #     # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
        #     clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
        #     condH.append(clusterscondH)
        #   #   barvals = []
        #   #   for c in classesPopDS[0]:
        #   #     if c in classesPop[0]:
        #   #       barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
        #   #     else:
        #   #       barvals.append(0)
        #   #   ticks = range(18)
        #   #   # gs = axs[i, -2].get_gridspec()
        #   #   # remove the underlying axes
        #   #   # for ax in axs[i, 10:]:
        #   #   #   ax.remove()
        #   #   # axbig = fig.add_subplot(gs[i, 10:])
        #   #   axs[k, 0].set_ylabel(
        #   #     'cluster no.{} \n H={:.4f},\n dominant class:{} \n p_class={:.4f}'.format(i, clusterscondH, np.argmax(population[i, :]), np.max(population[i, :])),
        #   #     rotation=0, size='medium', loc='bottom')
        #   #   for j in range(14):
        #   #     if j == 0:
        #   #       plt.setp(axs[k, j].spines.values(), color='white')
        #   #       axs[k, j].set_xticks([])
        #   #       axs[k, j].set_yticks([])
        #   #     elif j < 6:
        #   #       axs[k, j].imshow(ims_true[i][j - 1])
        #   #       axs[k, j].set_xticks([])
        #   #       axs[k, j].set_yticks([])
        #   #       # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
        #   #     elif j == 6:
        #   #       axs[k, j].remove()
        #   #     elif j < 12:
        #   #       try:
        #   #         axs[k, j].imshow(ims_false[i][j - 7])
        #   #         axs[k, j].set_xticks([])
        #   #         axs[k, j].set_yticks([])
        #   #         # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
        #   #       except:
        #   #         plt.setp(axs[k, j].spines.values(), color='white')
        #   #         axs[k, j].set_xticks([])
        #   #         axs[k, j].set_yticks([])
        #   #     elif j==12:
        #   #       # ticks = range(len(classesPop[1]))
        #   #       axs[k, j].bar(ticks, barvals, align='center')
        #   #       axs[k, j].set_xticks(ticks)
        #   #       axs[k, j].tick_params(axis='x', rotation=0)
        #   #       # axs[i,j].hist(GT_cluster, )
        #   #     else:
        #   #       axs[k, j].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
        #   #       axs[k, j].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)
        #   #
        #   # fig.align_ylabels()
        #   # # fig.tight_layout()
        #   # plt.savefig(
        #   #   'Thesis figures/pascal/clustering/noBG/{}_clusters_{}_eigvecs_{:.4f}_MI.png'.format(n, vLen, MI1),
        #   #   bbox_inches='tight')
        #   # plt.close(fig)
        #
        #   conHs[str(n) + ' ' + str(n)] = condH
        #   condHperClass = {}
        #   for m in range(len(classesPopDS[0])):
        #     Hs = []
        #     for i, c in enumerate(patches2Include):
        #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i,:])==np.flatnonzero(list(classesIdx.values())==p['seg']))]
        #       # if np.max(population[i,:]) >= purity:
        #       #   chance2classify[np.argmax(population[i,:])] += len(ps)/classesPopDS[1][np.argmax(population[i,:])]
        #       for p in ps:
        #         # print(type(classesIdx[m]))
        #         if p['seg'] == classesIdx[m]:
        #           Hs.append(condH[i])
        #     condHperClass[m] = Hs
        #   for pu in purity:
        #     chance2classify = np.zeros(len(classesPopDS[0]))
        #     for i, c in enumerate(patches2Include):
        #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i, :]) == np.flatnonzero(
        #         list(classesIdx.values()) == p['seg']))]
        #       if np.max(population[i, :]) >= pu:
        #         chance2classify[np.argmax(population[i, :])] += len(ps) / classesPopDS[1][np.argmax(population[i, :])]
        #     chance2cl[str(n)+'_'+str(n)+'_'+str(pu)] = chance2classify
        #   condHperC[str(n)+'_'+str(n)] = condHperClass
          for vLen in vLens:
            eigenValues, eigenVectors = eigsh(L_sym, k=n, which='SM')  # SM = Smallest Magnitude
            # Optional: sort if needed
            idx = np.argsort(eigenValues)
            eigenValues = eigenValues[idx]
            eigenVectors = eigenVectors[:, idx]
            purity_train = []
            purity_test = []
            train_label = []
            test_label = []
            vLen=n
            ims_true = []
            ims_false = []
            # space = eigenVectors[:, :vLen]
            space = eigenVectors
            normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
            Y = (space.T / normalize_f).T
            Kmeans = sklearn.cluster.KMeans(n_clusters=n, n_init=50)
            im_ids = random.sample(range(np.max([p['im_id'] for p in patches])),
                                   int(np.max([p['im_id'] for p in patches]) * 2/3))
            patches_fit = [p['idx'] for p in patches if p['im_id'] in im_ids]
            fit_sapce = Y[patches_fit, :]
            clusters = Kmeans.fit(fit_sapce)
            patches_predict = [p['idx'] for p in patches if p['im_id'] not in im_ids]
            distsFromClustersCenter = clusters.transform(Y)
            predict_space = Y[patches_predict, :]
            predictions = Kmeans.predict(predict_space)
            # temp = list(clusters.labels_)
            # for i, pred in zip(patches_predict, predictions):
            #   temp.insert(i, pred)
            # clusters.labels_ = np.array(temp)
            distsperCluster = [
              (distsFromClustersCenter[np.array(patches_fit)[np.where(clusters.labels_ == i)]], np.array(patches_fit)[np.where(clusters.labels_ == i)]) for i in
              range(n)]
            patches2Include = [distsperCluster[i][1][np.where(
              distsperCluster[i][0][:, i] <= np.percentile(distsperCluster[i][0][:, i], 100))[0]] for i in range(n)]
            # clusters_trans = Kmeans.fit_transform(Y)
            # clustersCenterD.append(np.mean(
            #   [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
            #    for j in range(n) if i != j]))
            # inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
          # # plot the change in the distances according to the size of the new patches descriptors
          #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
          #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
          #   ax2.set_xlabel('# eig vec')
          #   ax2.set_ylabel('mean distance')
          #   ax2.legend()
            # simulate and test the population of one cluster
            population = np.zeros((n, len(classesPopDS[0])))
            probs = np.zeros(population.shape)
            # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
            # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
            clustersNames = range(n)
            # labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
            #                'scissors background', 'airplane', 'airplane background']
            # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
            #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
            counter = np.zeros((1, n))
            for clusterIDX in range(n):
              # idxlist = np.where(clusters.labels_ == clusterIDX)
              idxlist = np.concatenate([patches2Include[clusterIDX], np.array(patches_predict)[np.where(predictions==clusterIDX)]])
              #       print(len(idxlist[0])/len(patches))
              gt_train = [(p['seg']) for p in patches if p['idx'] in patches2Include[clusterIDX]]
              gt_test = [(p['seg']) for p in patches if
                         p['idx'] in np.array(patches_predict)[np.where(predictions == clusterIDX)]]
              train_pop = np.unique(gt_train, return_counts=True, axis=0)
              test_pop = np.unique(gt_test, return_counts=True, axis=0)
              purity_train.append(np.max(train_pop[1])/sum(train_pop[1]))
              train_label.append(train_pop[0][np.argmax(train_pop[1])])
              try:
                purity_test.append(np.max(test_pop[1])/sum(test_pop[1]))
                test_label.append(test_pop[0][np.argmax(test_pop[1])])
              except:
                purity_test.append(None)
                test_label.append(None)
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist]
              classesPop = np.unique(gt_train, return_counts=True, axis=0) #np.unique(GT_cluster, return_counts=True, axis=0)
              max_Pclass = copy.deepcopy(classesPop[0])[np.argmax(classesPop[1])]
              for i, pop in enumerate(classesPop[0]):
                val = list(classesIdx.values()).index(pop)
                # population[clusterIDX, val] = classesPop[1][i] / len(idxlist[0])
                # if b'bus' in pop and b'255' in pop:
                #   val = 0
                # if b'bus' in pop and b'0' in pop:
                #   val = 1
                # if b'dog' in pop and b'255' in pop:
                #   val = 2
                # if b'dog' in pop and b'0' in pop:
                #   val = 3
                # if b'pizza' in pop and b'255' in pop:
                #   val = 4
                # if b'pizza' in pop and b'0' in pop:
                #   val = 5
                # if b'scissors' in pop and b'255' in pop:
                #   val = 6
                # if b'scissors' in pop and b'0' in pop:
                #   val = 7
                # if b'airplane' in pop and b'255' in pop:
                #   val = 8
                # if b'airplane' in pop and b'0' in pop:
                #   val = 9
                classesPop[0][i] = val
                population[clusterIDX, int(classesPop[0][i])] = classesPop[1][i] / sum(classesPop[1])
                probs[clusterIDX, int(classesPop[0][i])] = classesPop[1][i]/len(patches)
              # max_Pclass = classesPop[0][np.argmax(classesPop[1])]
              try:
                true_im = [p['ref'] for p in patches if
                           (p['idx'] in idxlist and max_Pclass == p['seg'])]
                ims_true.append(random.sample(true_im, k=np.min([5, len(true_im)])))
              except:
                print(max_Pclass)
              false_im = [p['ref'] for p in patches if (
                        p['idx'] in idxlist and max_Pclass != p['seg'])]
              try:
                ims_false.append(random.sample(false_im, k=5))
              except:
                ims_false.append(false_im)
              counter[0, clusterIDX] = np.sum(classesPop[1])
            # #       for i, pop in enumerate(classesPop[0]):
            # #         if b'elephant' in pop and b'255' in pop:
            # #           val = 4
            # #         if b'elephant' in pop and b'0' in pop:
            # #           val = 5
            # #         if b'dog' in pop and b'255' in pop:
            # #           val = 2
            # #         if b'dog' in pop and b'0' in pop:
            # #           val = 3
            # #         if b'fire hydrant' in pop and b'255' in pop:
            # #           val = 6
            # #         if b'fire hydrant' in pop and b'0' in pop:
            # #           val = 7
            # #         if b'train' in pop and b'255' in pop:
            # #           val = 8
            # #         if b'train' in pop and b'0' in pop:
            # #           val = 9
            # #         if b'airplane' in pop and b'255' in pop:
            # #           val = 0
            # #         if b'airplane' in pop and b'0' in pop:
            # #           val = 1
            # #         classesPop[0][i] = val
            # #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
            # #         counter[val] += 1
            MI1 = sklearn.metrics.mutual_info_score([p['seg'] for p in patches if p['idx'] in np.hstack(patches2Include)], clusters.labels_)#[clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)])
            MI_n.append(MI1)
            clusters2show = {}

            sortbyH = [sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0]) for i in range(n)]
            sortbyclass = [np.argmax(population[i, :]) for i in range(n)]
            sortedclustersIdx = np.lexsort((sortbyH, sortbyclass))
            # create similarity hists
            patchesBycluster = np.hstack(patches2Include)
            # distsFromClustersCenter = clusters.transform(Y)
            # distsperCluster = [distsFromClustersCenter[np.where(clusters.labels_ == i)] for i in range(n)]
            for k in range(n):
              i = sortedclustersIdx[k]
              if str(np.argmax(population[i, :])) in clusters2show.keys():
                continue
                # if clusters2show[str(np.argmax(population[i, :]))]==2:
                #   continue
                # else:
                #   clusters2show[str(np.argmax(population[i, :]))] = 2
              else:
                clusters2show[str(np.argmax(population[i, :]))]=1
            numOfRows = sum(clusters2show.values())+5
            fig, axs = plt.subplots(numOfRows, 15, figsize=(numOfRows, 15),
                                    gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 0.2, 2]})
            plt.subplots_adjust(hspace=0.4)
            # fig.suptitle('{} clusters, MI = {}'.format(n, MI1))
            # plt.subplots_adjust(wspace=0.05, hspace=0.6)  # , left=2, right=2.2)

            # dis4Hist = np.zeros(dissim.shape)
            # clustersSize = np.unique([clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)], return_counts=True)
            # for i in range(len(patchesBycluster)):
            #   dis4Hist[i, :] = dissim[patchesBycluster[i], :].toarray()
            # dissim_forcopy = copy.deepcopy(dis4Hist)
            # for i in range(len(patchesBycluster)):
            #   dis4Hist[:, i] = dissim_forcopy[:, patchesBycluster[i]]
            # dis4Hist = dis4Hist[:np.hstack(patches2Include).shape[0], :np.hstack(patches2Include).shape[0]]
            # dists_within_cluster = []
            # for i in range(n):
            #   if i == 0:
            #     dists_within_cluster.append(dis4Hist[:clustersSize[1][i], :clustersSize[1][i]])
            #   # elif i == n-1:
            #   #   dists_within_cluster.append(dis4Hist[-clustersSize[1][i]:, -clustersSize[1][i]:])
            #   else:
            #     dists_within_cluster.append(dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
            #                               sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)])])
            #
            # dists_between_cluster = []
            # for i in range(n):
            #   if i == 0:
            #     dists_between_cluster.append(dis4Hist[:clustersSize[1][i], clustersSize[1][i]:])
            #   # elif i == n-1:
            #   #   dists_between_cluster.append(dis4Hist[-clustersSize[1][i]:, :-clustersSize[1][i]])
            #   else:
            #     dists_between_cluster.append(np.hstack((dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
            #                                           :sum(clustersSize[1][:i])],
            #                                           dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
            #                                           sum(clustersSize[1][:(i + 1)]):])))
            clusters2show = {}
            condH = []
          ### old plots start here
          #   for k in range(n):
          #     i = sortedclustersIdx[k]
          #     idxlist = np.where(clusters.labels_ == i)
          #     #       print(len(idxlist[0])/len(patches))
          #     GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
          #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
          #     # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
          #     clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
          #     condH.append(clusterscondH)
          #   #   barvals = []
          #   #   for c in classesPopDS[0]:
          #   #     if c in classesPop[0]:
          #   #       barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
          #   #     else:
          #   #       barvals.append(0)
          #   #   ticks = range(18)
          #   #   # gs = axs[i, -2].get_gridspec()
          #   #   # remove the underlying axes
          #   #   # for ax in axs[i, 10:]:
          #   #   #   ax.remove()
          #   #   # axbig = fig.add_subplot(gs[i, 10:])
          #   #   axs[k, 0].set_ylabel(
          #   #     'cluster no.{} \n H={:.4f},\n dominant class:{} \n p_class={:.4f}'.format(i, clusterscondH, np.argmax(population[i, :]), np.max(population[i, :])),
          #   #     rotation=0, size='medium', loc='bottom')
          #   #   for j in range(14):
          #   #     if j == 0:
          #   #       plt.setp(axs[k, j].spines.values(), color='white')
          #   #       axs[k, j].set_xticks([])
          #   #       axs[k, j].set_yticks([])
          #   #     elif j < 6:
          #   #       axs[k, j].imshow(ims_true[i][j - 1])
          #   #       axs[k, j].set_xticks([])
          #   #       axs[k, j].set_yticks([])
          #   #       # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
          #   #     elif j == 6:
          #   #       axs[k, j].remove()
          #   #     elif j < 12:
          #   #       try:
          #   #         axs[k, j].imshow(ims_false[i][j - 7])
          #   #         axs[k, j].set_xticks([])
          #   #         axs[k, j].set_yticks([])
          #   #         # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
          #   #       except:
          #   #         plt.setp(axs[k, j].spines.values(), color='white')
          #   #         axs[k, j].set_xticks([])
          #   #         axs[k, j].set_yticks([])
          #   #     elif j==12:
          #   #       # ticks = range(len(classesPop[1]))
          #   #       axs[k, j].bar(ticks, barvals, align='center')
          #   #       axs[k, j].set_xticks(ticks)
          #   #       axs[k, j].tick_params(axis='x', rotation=0)
          #   #       # axs[i,j].hist(GT_cluster, )
          #   #     else:
          #   #       axs[k, j].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
          #   #       axs[k, j].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)
          #   #
          #   # fig.align_ylabels()
          #   # # fig.tight_layout()
          #   # plt.savefig(
          #   #   'Thesis figures/pascal/clustering/noBG/{}_clusters_{}_eigvecs_{:.4f}_MI.png'.format(n, vLen, MI1),
          #   #   bbox_inches='tight')
          #   # plt.close(fig)
          #
          #   conHs[str(n) + ' ' + str(vLen)] = condH
          #   condHperClass = {}
          #   for m in range(len(classesPopDS[0])):
          #     Hs = []
          #     for i, c in enumerate(patches2Include):
          #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i,:])==np.flatnonzero(list(classesIdx.values())==p['seg']))]
          #       # if np.max(population[i,:]) >= purity:
          #       #   chance2classify[np.argmax(population[i,:])] += len(ps)/classesPopDS[1][np.argmax(population[i,:])]
          #       for p in ps:
          #         # print(type(classesIdx[m]))
          #         if p['seg'] == classesIdx[m]:
          #           Hs.append(condH[i])
          #     condHperClass[m] = Hs
          #   for pu in purity:
          #     chance2classify = np.zeros(len(classesPopDS[0]))
          #     cov = np.zeros(len(classesPopDS[0]))
          #     for i, c in enumerate(patches2Include):
          #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i, :]) == np.flatnonzero(
          #         list(classesIdx.values()) == p['seg']))]
          #       if np.max(population[i, :]) >= pu:
          #         chance2classify[np.argmax(population[i, :])] += len(ps) / classesPopDS[1][np.argmax(population[i, :])]
          #         cov[np.argmax(population[i, :])] += len(ps)
          #     chance2cl[str(n)+'_'+str(pu)] = chance2classify
          #     coverage[str(vLen)+'_'+str(pu)] = cov
          #   condHperC[str(n)+'_'+str(vLen)] = condHperClass
          # MIs[str(n)] = MI_n
            for k in range(n):
              ind = 0
              if clusters2show.values():
                ind = sum(clusters2show.values())
              i = sortedclustersIdx[k]

              idxlist = np.concatenate([np.array(patches_fit)[np.where(clusters.labels_ == i)], np.array(patches_predict)[np.where(predictions==i)]])
              #       print(len(idxlist[0])/len(patches))
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist]
              gt_train = [(p['seg']) for p in patches if p['idx'] in np.array(patches_fit)[np.where(clusters.labels_ == i)]]
              gt_test = [(p['seg']) for p in patches if
                         p['idx'] in np.array(patches_predict)[np.where(predictions == i)]]
              classesPop = np.unique(gt_train, return_counts=True, axis=0)
              # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
              clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
              condH.append(clusterscondH)
              if str(np.argmax(population[i, :])) in clusters2show.keys():
                continue
                # if clusters2show[str(np.argmax(population[i, :]))]==2:
                #   continue
                # else:
                #   clusters2show[str(np.argmax(population[i, :]))] = 2
              else:
                clusters2show[str(np.argmax(population[i, :]))]=1
              barvals = []
              for c in classesPopDS[0]:
                if c in classesPop[0]:
                  barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
                else:
                  barvals.append(0)
            #   ticks = range(18)
            #   # gs = axs[i, -2].get_gridspec()
            #   # remove the underlying axes
            #   # for ax in axs[i, 10:]:
            #   #   ax.remove()
            #   # axbig = fig.add_subplot(gs[i, 10:])
              axs[ind, 0].set_ylabel(
                'cluster no.{} \n H={:.4f} #im={}'.format(i, clusterscondH, len(np.unique([(p['im_id']) for p in patches if p['idx'] in idxlist], return_counts=True)[1])),
                rotation=0, size='medium', loc='center')
              axs[ind, 1].set_ylabel(
                'dominant train:{} \n purity={:.4f} \n dominant test:{} \n purity={:.4f}'.format(train_label[i],
                                                                                                 purity_train[i],
                                                                                                 test_label[i],
                                                                                                 purity_test[i]),
                rotation=0, size='small', loc='center')
              for j in range(15):
                if j == 0 or j==1:
                  plt.setp(axs[ind, j].spines.values(), color='white')
                  axs[ind, j].set_xticks([])
                  axs[ind, j].set_yticks([])
                elif j < 7:
                  try:
                    axs[ind, j].imshow(ims_true[i][j - 2])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                  # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
                elif j == 7:
                  axs[ind, j].remove()
                elif j < 13:
                  try:
                    axs[ind, j].imshow(ims_false[i][j - 8])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                    # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                elif j==13:
                  axs[ind, j].remove()
                elif j < 15:
                  ticks = range(len(barvals))
                  axs[ind, j].bar(ticks, barvals, align='center')
                  # axs[ind, j].set_xticks(ticks)
                  axs[ind, j].tick_params(axis='x', rotation=0)
                  # axs[i,j].hist(GT_cluster, )
                #   axs[ind, j+1].remove()
                #   axs[ind, j+2].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
                #   axs[ind, j+2].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)

            for ind, k in enumerate(np.argsort(sortbyH)[-5:]):

              i = k
              ind += sum(clusters2show.values())
              idxlist = np.where(clusters.labels_ == i)[0]
              #       print(len(idxlist[0])/len(patches))
              print(idxlist)
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist]
              classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
              # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
              clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
              condH.append(clusterscondH)
              barvals = []
              for c in classesPopDS[0]:
                if c in classesPop[0]:
                  barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
                else:
                  barvals.append(0)
            #   ticks = range(18)
            #   # gs = axs[i, -2].get_gridspec()
            #   # remove the underlying axes
            #   # for ax in axs[i, 10:]:
            #   #   ax.remove()
            #   # axbig = fig.add_subplot(gs[i, 10:])
              axs[ind, 0].set_ylabel(
                'cluster no.{} \n H={:.4f} #im={}' .format(i, clusterscondH, len(np.unique([(p['im_id']) for p in patches if p['idx'] in idxlist], return_counts=True)[1])),
                rotation=0, size='small', loc='center')
              axs[ind, 1].set_ylabel(
                'dominant train:{} \n purity={:.4f} \n dominant test:{} \n purity={:.4f}'.format(train_label[i], purity_train[i], test_label[i], purity_test[i]),
                rotation=0, size='small', loc='center')
              for j in range(15):
                if j == 0 or j==1:
                  plt.setp(axs[ind, j].spines.values(), color='white')
                  axs[ind, j].set_xticks([])
                  axs[ind, j].set_yticks([])
                elif j < 7:
                  try:
                    axs[ind, j].imshow(ims_true[i][j - 2])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                  # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
                elif j == 7:
                  axs[ind, j].remove()
                elif j < 13:
                  try:
                    axs[ind, j].imshow(ims_false[i][j - 8])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                    # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                elif j == 13:
                  axs[ind, j].remove()
                elif j<15:
                  ticks = range(len(barvals))
                  axs[ind, j].bar(ticks, barvals, align='center')
                  # axs[ind, j].set_xticks(ticks)
                  axs[ind, j].tick_params(axis='x', rotation=0)
                  # axs[i,j].hist(GT_cluster, )
                #   axs[ind, j+1].remove()
                #   axs[ind, j+2].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
                #   axs[ind, j+2].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)
            fig.align_ylabels()
            # fig.tight_layout()
            plt.savefig(
              'Thesis figures/pascal/samples/largeDS_{}_clusters_{:.4f}_MI.png'.format(n, MI1),
              bbox_inches='tight')
            # plt.show()
            plt.close(fig)

            conHs[str(n) + ' ' + str(vLen)] = condH
            condHperClass = {}
            for m in range(len(classesPopDS[0])):
              Hs = []
              for i, c in enumerate(patches2Include):
                ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i,:])==list(classesIdx.values()).index(p['seg']))]
                # if np.max(population[i,:]) >= purity:
                #   chance2classify[np.argmax(population[i,:])] += len(ps)/classesPopDS[1][np.argmax(population[i,:])]
                for p in ps:
                  # print(type(classesIdx[m]))
                  if p['seg'] == classesIdx[m]:
                    Hs.append(condH[i])
              condHperClass[m] = Hs
            for pu in purity:
              chance2classify = np.zeros(len(classesPopDS[0]))
              cov = np.zeros(len(classesPopDS[0]))
              for i, c in enumerate(patches2Include):
                ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i, :]) == list(classesIdx.values()).index(p['seg']))]
                if np.max(population[i, :]) >= pu:
                  chance2classify[np.argmax(population[i, :])] += len(ps) / classesPopDS[1][np.argmax(population[i, :])]
                  cov[np.argmax(population[i, :])] += len(ps)
              chance2cl[str(n)+'_'+str(pu)] = chance2classify
              coverage[str(vLen)+'_'+str(pu)] = cov
            condHperC[str(n)+'_'+str(vLen)] = condHperClass
          MIs[str(n)] = MI_n



            # plt.show()
          # # simulate and test the population of one cluster
          #   population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
          #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
          #   clustersNames = range(n)
          #   labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background','elephant',
          #   'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
          #   for clusterIDX in range(n):
          #     idxlist = np.where(clusters.labels_ == clusterIDX)
          #     print(len(idxlist)/len(patches))
          #     GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
          #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
          #     for i, pop in enumerate(classesPop[0]):
          #       if b'airplane' in pop and b'255' in pop:
          #         classesPop[0][i] = 0
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'airplane' in pop and b'0' in pop:
          #         classesPop[0][i] = 1
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'dog' in pop and b'255' in pop:
          #         classesPop[0][i] = 2
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'dog' in pop and b'0' in pop:
          #         classesPop[0][i] = 3
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'fire hydrant' in pop and b'255' in pop:
          #         classesPop[0][i] = 6
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'fire hydrant' in pop and b'0' in pop:
          #         classesPop[0][i] = 7
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'elephant' in pop and b'255' in pop:
          #         classesPop[0][i] = 4
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'elephant' in pop and b'0' in pop:
          #         classesPop[0][i] = 5
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'train' in pop and b'255' in pop:
          #         classesPop[0][i] = 8
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'train' in pop and b'0' in pop:
          #         classesPop[0][i] = 9
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #   fig, ax = plt.subplots()
          #   im = ax.imshow(population)
          #   ax.set_yticks(np.arange(len(clustersNames)))
          #   ax.set_xticks(np.arange(len(labelsNames)))
          #   cb = plt.colorbar(im)
          #   plt.savefig('heatmaps_coco_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
          #   plt.close(fig)
            # population[clusterIDX, classesPop[0]] = classesPop[1]
            # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
            #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])

          # plt.show()
          # plt.savefig('keepDaway_newpatches.png')
          # plt.close(fig2)
          # fig, ax = plt.subplots()
          # ax.imshow(population)
          # ax.set_yticks(np.arange(len(clustersNames)), labels=clustersNames)
          # ax.set_xticks(np.arange(len(labelsNames)), labels=labelsNames)
        ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
        random.shuffle(ims)
        fig, axs = plt.subplots(6, 6)
        for i in range(6):
          for j in range(6):
            axs[i, j].imshow(ims[i * 6 + j])
        plt.show()
          # patches = sorted(patches, key=lambda p: p['idx'])
          # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
          # clustering_kmeans = SpectralClustering(n_clusters=50,
          #                                        assign_labels='kmeans',
          #                                        random_state=0,
          #                                        affinity='nearest_neighbors').fit(
          #   np.array([p['patch'].flatten() for p in patches]))
          # clustering_disc = SpectralClustering(n_clusters=50,
          #                                      assign_labels='discretize',
          #                                      random_state=0,
          #                                      affinity='nearest_neighbors').fit(
          #   np.array([p['patch'].flatten() for p in patches]))
          #
          # plt.hist(clustering_disc.labels_, bins=50)
          # plt.hist(clustering_kmeans.labels_, bins=50)


      # except Exception as e:  # pylint: disable=broad-except
      #   coord.request_stop(e)
      #
      # coord.request_stop()
      # coord.join(threads, stop_grace_period_secs=10)
  elif coco:
    with open('patches_8_coco_pca24_pascal.pkl', 'rb') as f:
      patches = pickle.load(f)
        # patches = random.choices(patches, k=500)
        # threads_per_block = (16,16)
        # blocks_per_grid = ((4224+16-1)//16, (4224+16-1)//16)
        # patches_array = [p['patch'] for p in patches]
        # patches_cuda = cuda.to_device(patches_array)

      # # # # dissim = calcPatchDistmulti(doues)

      # # # device = 'cuda'
      # #   # dissim = cuda.to_device(dissim)
      # #   # dissim = calcDistGPUPar[blocks_per_grid,threads_per_block](patches_cuda, dissim)
      # doues = comb(patches, 2)
      # dissim = np.zeros((len(patches), len(patches)))
      # with multiprocessing.Pool(processes=16) as p:
      #   with tqdm(total=(len(patches) ** 2)/2) as pbar:
      #     for d in p.imap_unordered(calcPatchDistmulti, doues):
      #       pbar.update()
      #       dissim[d[0], d[1]] = d[2]
      #       dissim[d[1], d[0]] = d[2]
      # # #   # # for p1 in tqdm(patches):
      # # #   # #   for p2 in patches:
      # # #   # #     if dissim[p1['idx'], p2['idx']] != 0:
      # # #   # #       continue
      # # #   # #     else:
      # # #   # #       # start = time.time()
      # # #   # #       # patch1 = torch.from_numpy(p1['patch'])
      # # #   # #       # patch2 = torch.from_numpy(p2['patch'])
      # # #   # #       # dist = calcPatchDisttorch(patch1, patch2)
      # # #   # #       dist = calcPatchDist(p1['patch'], p2['patch'])
      # # #   # #       dissim[p1['idx'], p2['idx']] = dist
      # # #   # #       dissim[p2['idx'], p1['idx']] = dist
      # # #   # #       # print(time.time()-start)
      # with open('dissim_8_coco_pca16_pascal_{}.pkl'.format(datetime.now()), 'wb') as f:
      #   pickle.dump(dissim, f)
    with open('dissim_8_coco_pca16_pascal_2023-05-12 14:11:06.126746.pkl', 'rb') as f:
      dissim = pickle.load(f)
    max_d = np.amax(dissim)
    # dissim = 1 - (dissim / max_d)
    sigma = np.percentile(dissim, sigP)
    affinity = np.exp(-dissim/sigma)
    D = np.zeros(affinity.shape)
    for i in range(affinity.shape[0]):
      D[i, i] = sum(affinity[i, :])
    L = np.matmul(np.matmul(np.diag(np.diag(D) ** (-1 / 2)), affinity), np.diag(np.diag(D) ** (-1 / 2)))
    eig = np.linalg.eig(L)
    vLens = range(5, 51, 5)
    # vLens = [10]
    clustersCenterD = []
    inClustersD = []
    ns = range(10,101,5)
    # ns = [100]
    fig2, ax2 = plt.subplots()
    limitV_n = []

    for n in ns:
      for vLen in vLens:
        ims_true = []
        ims_false = []
        space = eig[1][:, :vLen]
        normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
        Y = (space.T / normalize_f).T
        Kmeans = sklearn.cluster.KMeans(n_clusters=n, algorithm='elkan')
        clusters = Kmeans.fit(Y)
        clusters_trans = Kmeans.fit_transform(Y)
        clustersCenterD.append(np.mean(
          [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
           for j in range(n) if i != j]))
        inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
      # idxlimit = np.argwhere(np.diff(np.sign(np.array(clustersCenterD[-len(vLens):])- np.array(inClustersD[-len(vLens):])))).flatten()
      # if idxlimit:
      #   limitV_n.append((n, vLens[idxlimit[0]]))
    # # plot the change in the distances according to the size of the new patches descriptors
    #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between clusters mean distance n={}'.format(n))
    #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
    #   ax2.set_xlabel('# eig vec')
    #   ax2.set_ylabel('mean distance')
    #   ax2.legend(fontsize='x-small')

    # simulate and test the population of one cluster
        population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
        # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
        clustersNames = range(n)
        labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
                       'scissors background', 'airplane', 'airplane background']
        # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
        #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
        counter = np.zeros((1, n))
        for clusterIDX in range(n):
          idxlist = np.where(clusters.labels_ == clusterIDX)
    #       print(len(idxlist[0])/len(patches))
          GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
          classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
          max_Pclass = copy.deepcopy(classesPop[0])[np.argmax(classesPop[1])]
          for i, pop in enumerate(classesPop[0]):
            # val = [item[0] for item in idxclasses if item[1] == pop][0]
            # population[clusterIDX, val] = classesPop[1][i] / len(idxlist[0])
            if b'bus' in pop and b'255' in pop:
              val = 0
            if b'bus' in pop and b'0' in pop:
              val = 1
            if b'dog' in pop and b'255' in pop:
              val = 2
            if b'dog' in pop and b'0' in pop:
              val = 3
            if b'pizza' in pop and b'255' in pop:
              val = 4
            if b'pizza' in pop and b'0' in pop:
              val = 5
            if b'scissors' in pop and b'255' in pop:
              val = 6
            if b'scissors' in pop and b'0' in pop:
              val = 7
            if b'airplane' in pop and b'255' in pop:
              val = 8
            if b'airplane' in pop and b'0' in pop:
              val = 9
            classesPop[0][i] = val
            population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]/sum(classesPop[1])
          # max_Pclass = classesPop[0][np.argmax(classesPop[1])]
          try:
            true_im = [p['ref'] for p in patches if (p['idx'] in idxlist[0] and max_Pclass[0] in p['label'] and int(max_Pclass[1]) == p['seg'])]
            ims_true.append(random.sample(true_im, k=5))
          except:
            print(max_Pclass)
          false_im = [p['ref'] for p in patches if (p['idx'] in idxlist[0] and (max_Pclass[0] not in p['label'] or int(max_Pclass[1]) != p['seg']))]
          try:
            ims_false.append(random.sample(false_im, k=5))
          except:
            ims_false.append(false_im)
          counter[0,clusterIDX] = np.sum(classesPop[1])
    # #       for i, pop in enumerate(classesPop[0]):
    # #         if b'elephant' in pop and b'255' in pop:
    # #           val = 4
    # #         if b'elephant' in pop and b'0' in pop:
    # #           val = 5
    # #         if b'dog' in pop and b'255' in pop:
    # #           val = 2
    # #         if b'dog' in pop and b'0' in pop:
    # #           val = 3
    # #         if b'fire hydrant' in pop and b'255' in pop:
    # #           val = 6
    # #         if b'fire hydrant' in pop and b'0' in pop:
    # #           val = 7
    # #         if b'train' in pop and b'255' in pop:
    # #           val = 8
    # #         if b'train' in pop and b'0' in pop:
    # #           val = 9
    # #         if b'airplane' in pop and b'255' in pop:
    # #           val = 0
    # #         if b'airplane' in pop and b'0' in pop:
    # #           val = 1
    # #         classesPop[0][i] = val
    # #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
    # #         counter[val] += 1
        fig, ax = plt.subplots(figsize=(6, 20))
        # counter = np.broadcast_to(counter.T, population.shape)
        # population = np.divide(population, counter)
        im = ax.imshow(population)
        ax.set_yticks(np.arange(len(clustersNames)))
        ax.set_xticks(np.arange(len(labelsNames)))
        # for y in range(n):
        #   for x in range(10):
        #     label = population[x, y]
        #     text_x = x
        #     text_y = y
        #     ax.text(text_x, text_y, label, color='black', ha='center', va='center')
        cb = plt.colorbar(im)
        # plt.show()
        plt.savefig('heatmaps_coco_clustering/pascalweights/purity/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
        plt.close(fig)
        ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        fig, axs = plt.subplots(n, 12, figsize=(20, 120))
        for i in range(n):
          idxlist = np.where(clusters.labels_ == i)
          #       print(len(idxlist[0])/len(patches))
          GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
          classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
          # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
          ticks = [str(classesPop[0][i][0]) + bool(1-int(classesPop[0][i][1])/255)*' background' for i in range(len(classesPop[0]))]
          gs = axs[i, -2].get_gridspec()
          # remove the underlying axes
          for ax in axs[i, 10:]:
            ax.remove()
          axbig = fig.add_subplot(gs[i, 10:])
          for j in range(11):
            if j<5:
              axs[i, j].imshow(ims_true[i][j]+ds_mean)
              # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
            elif j<10:
              try:
                axs[i, j].imshow(ims_false[i][j - 5]+ds_mean)
                # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
              except:
                continue
            else:
              # ticks = range(len(classesPop[1]))
              axbig.bar(ticks, classesPop[1], align='center')
              axbig.set_xticks(ticks)
              axbig.tick_params(axis='x', rotation=15)
              # axs[i,j].hist(GT_cluster, )
        fig.tight_layout()
        plt.savefig(
          'heatmaps_coco_clustering/pascalweights/samples/imgandhist_{}_clusters_{}_eigvecs.png'.format(n, vLen))
        plt.close(fig)
        # plt.show()
      # population[clusterIDX, classesPop[0]] = classesPop[1]
      # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
      #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])
    #
    # plt.show()
    # plt.savefig('keepDaway_percentile2_pca16.png' ,dpi=600)
    # plt.close(fig2)
    # fig, ax = plt.subplots()
    # ax.imshow(population)
    # ax.set_yticks(np.arange(len(clustersNames)))
    # ax.set_xticks(np.arange(len(labelsNames)))
    ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
    random.shuffle(ims)
    fig, axs = plt.subplots(6, 6)
    for i in range(6):
      for j in range(6):
        axs[i, j].imshow(ims[i * 6 + j])
    plt.show()
    # patches = sorted(patches, key=lambda p: p['idx'])
    # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
    # clustering_kmeans = SpectralClustering(n_clusters=50,
    #                                        assign_labels='kmeans',
    #                                        random_state=0,
    #                                        affinity='nearest_neighbors').fit(
    #   np.array([p['patch'].flatten() for p in patches]))
    # clustering_disc = SpectralClustering(n_clusters=50,
    #                                      assign_labels='discretize',
    #                                      random_state=0,
    #                                      affinity='nearest_neighbors').fit(
    #   np.array([p['patch'].flatten() for p in patches]))
    #
    # plt.hist(clustering_disc.labels_, bins=50)
    # plt.hist(clustering_kmeans.labels_, bins=50)
  elif Rect:
    image = getRep_Rect()
    _, _, fuse = resnet50.inference(image, tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + 1,
       resnet50_input.NUM_CLASSES]), tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + 1,
       resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1)) as sess:
      sess.run(tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables()))
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
                                           start=True))

        patches = getPatchDS_rect(sess, image, fuse, patch_size=8, ifPCA=True, n=3, filenames=None)
        a=1

      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  elif MINC:
    preproccessMINC()
    image, label = getRep_MINC()

    _, _, fuse = resnet50.inference(image, tf.zeros(
      [1, image.shape[1] //4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]), tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1)) as sess:
      sess.run(tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables()))
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
                                           start=True))
        # patches = getPatchDS_MINC(sess, image, label, fuse, patch_size=8, ifPCA=True, n=16)
        # patches = getPatchDS_frompkl()

        # with open('patches_MINC_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(patches, f)
        # scores = dist_methods_comparison(patches)
        with open('patches_MINC_2025-06-29 12:05:22.556887.pkl', 'rb') as f:
          patches = pickle.load(f)

        prs = range(0, 101, 10)
        ims = range(150)
        scores = np.zeros((11, 150))
        for im in tqdm(ims):
          for i, pr in enumerate(prs):
            try:
              b, w, c = dists_im(patches,im, pr)
            except:
              break
            scores[i,im] += score(w, b)
        scores /= 150
        # lda = LDA(new_patches)
  #     # dissim =
  # #
  # # # # #       # patches = random.choices(patches, k=500)
  # # # # #       # threads_per_block = (16,16)
  # # # # #       # blocks_per_grid = ((4224+16-1)//16, (4224+16-1)//16)
  # # # # #       # patches_array = [p['patch'] for p in patches]
  # # # # #       # patches_cuda = cuda.to_device(patches_array)
  #       doues = comb(new_patches, 2)
  # #       # doues = [(d, lda) for d in doues]
  # # # # # # # # #       # dissim = calcPatchDistmulti(doues)
  #       data = []
  #       labels = []
  # #       # p1order = []
  # #       # p2order = []
  #       with multiprocessing.Pool(processes=16) as p:
  #           with tqdm(total=(len(new_patches) ** 2)/2) as pbar:
  #             for out in p.imap_unordered(calcPatchDist4lda, doues):
  #               pbar.update()
  #               _, _, d, l = out
  #               # p1 = np.where([p['idx']==p1 for p in new_patches])
  #               # p2 = np.where([p['idx']==p2 for p in new_patches])
  #               # p1order.append(p1)
  #               # p2order.append(p2)
  #               data.append(d)
  #               labels.append(int(l))
  #       data = np.array(data)
  #       labels = np.array(labels)
  #       weights = lda_fromScratch(data, labels)
        doues = comb(patches, 2)
        dissim = np.zeros((len(patches), len(patches)))
  #       with multiprocessing.Pool(processes=16) as p:
  #         with tqdm(total=(len(patches) ** 2) / 2) as pbar:
  #           for d in p.imap_unordered(calcPatchDist4lda, doues):
  #             pbar.update()
  #             p1, p2, dist, l = d
  #             dissim[p1,p2] = dissim[p2,p1] = np.array(dist).dot(weights)
  #       # with multiprocessing.Pool(processes=16) as p:
  #       #     with tqdm(total=(len(patches) ** 2)/2) as pbar:
  #       #       for d in p.imap_unordered(transorm2LDA, doues):
  #       #         pbar.update()
  #       #         dissim[d[0], d[1]] = d[2]
  #       #         dissim[d[1], d[0]] = d[2]
  # # # # #       # dissim = cuda.to_device(dissim)
  # # # # #       # dissim = calcDistGPUPar[blocks_per_grid,threads_per_block](patches_cuda, dissim)
        with multiprocessing.Pool(processes=16) as p:
          with tqdm(total=(len(patches) ** 2)/2) as pbar:
            for d in p.imap_unordered(calcPatchDistmulti, doues):
              pbar.update()
              dissim[d[0], d[1]] = d[2]
              dissim[d[1], d[0]] = d[2]
  # # # #       # # for p1 in tqdm(patches):
  # # # #       # #   for p2 in patches:
  # # # #       # #     if dissim[p1['idx'], p2['idx']] != 0:
  # # # #       # #       continue
  # # # #       # #     else:
  # # # #       # #       # start = time.time()
  # # # #       # #       # patch1 = torch.from_numpy(p1['patch'])
  # # # #       # #       # patch2 = torch.from_numpy(p2['patch'])
  # # # #       # #       # dist = calcPatchDisttorch(patch1, patch2)
  # # # #       # #       dist = calcPatchDist(p1['patch'], p2['patch'])
  # # # #       # #       dissim[p1['idx'], p2['idx']] = dist
  # # # #       # #       dissim[p2['idx'], p1['idx']] = dist
  # # # #       # #       # print(time.time()-start)
        with open('dissim_8_pascal_avg_filtered_q100.pkl', 'wb') as f:
          pickle.dump(dissim, f)
        with open('dissim_8_pascal_LDA_filtered_pinv.pkl', 'rb') as f:
          dissim_lda_svd = pickle.load(f)
        with open('dissim_8_pascal_LDA_filtered_diag.pkl', 'rb') as f:
          dissim_lda_diag = pickle.load(f)
        with open('dissim_8_pascal_pca16_filtered_percentile50.pkl', 'rb') as f:
          dissim_heu = pickle.load(f)
        with open('dissim_8_pascal_LDA_filtered_shrinkage.pkl', 'rb') as f:
          dissim_lda_convex = pickle.load(f)
        max_d = np.amax(dissim)
        affinity = 1 - (dissim / max_d)
        sigma = np.percentile(dissim, 2)
        affinity = np.exp(-dissim/sigma)
        D = np.zeros(affinity.shape)
        for i in range(affinity.shape[0]):
          D[i, i] = sum(affinity[i, :])
        L = np.matmul(np.matmul(np.diag(np.diag(D) ** (-1 / 2)), affinity), np.diag(np.diag(D) ** (-1 / 2)))
        eig = np.linalg.eig(L)
        # vLens = range(10, 101, 10)
        vLens = [50]
        clustersCenterD = []
        inClustersD = []
        # ns = range(60,121,10)
        ns = [80]
        fig2, ax2 = plt.subplots()
        segvals = [(p['seg']) for p in patches]
        classesPopDS = np.unique(segvals, return_counts=True, axis=0)
        classesIdx = {i: classesPopDS[0][i] for i in range(classesPopDS[0].shape[0])}
        for n in ns:
          for vLen in vLens:
            ims_true = []
            ims_false = []
            space = eig[1][:, :vLen]
            normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
            Y = (space.T / normalize_f).T
            Kmeans = sklearn.cluster.KMeans(n_clusters=n, algorithm='elkan')
            clusters = Kmeans.fit(Y)
            distsFromClustersCenter = clusters.transform(Y)
            distsperCluster = [
              (distsFromClustersCenter[np.where(clusters.labels_ == i)], np.where(clusters.labels_ == i)) for i in
              range(n)]
            patches2Include = [distsperCluster[i][1][0][np.where(
              distsperCluster[i][0][:, i] <= np.percentile(distsperCluster[i][0][:, i], 80))[0]] for i in range(n)]
            # clusters_trans = Kmeans.fit_transform(Y)
            # clustersCenterD.append(np.mean(
            #   [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
            #    for j in range(n) if i != j]))
            # inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
          # # plot the change in the distances according to the size of the new patches descriptors
          #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
          #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
          #   ax2.set_xlabel('# eig vec')
          #   ax2.set_ylabel('mean distance')
          #   ax2.legend()
            # simulate and test the population of one cluster
            population = np.zeros((n, len(classesPopDS[0])))
            probs = np.zeros(population.shape)
            # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
            # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
            clustersNames = range(n)
            # labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
            #                'scissors background', 'airplane', 'airplane background']
            # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
            #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
            counter = np.zeros((1, n))
            for clusterIDX in range(n):
              # idxlist = np.where(clusters.labels_ == clusterIDX)
              idxlist = [patches2Include[clusterIDX]]
              #       print(len(idxlist[0])/len(patches))
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
              classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
              max_Pclass = copy.deepcopy(classesPop[0])[np.argmax(classesPop[1])]
              for i, pop in enumerate(classesPop[0]):
                val = list(classesIdx.values()).index(pop)
                # population[clusterIDX, val] = classesPop[1][i] / len(idxlist[0])
                # if b'bus' in pop and b'255' in pop:
                #   val = 0
                # if b'bus' in pop and b'0' in pop:
                #   val = 1
                # if b'dog' in pop and b'255' in pop:
                #   val = 2
                # if b'dog' in pop and b'0' in pop:
                #   val = 3
                # if b'pizza' in pop and b'255' in pop:
                #   val = 4
                # if b'pizza' in pop and b'0' in pop:
                #   val = 5
                # if b'scissors' in pop and b'255' in pop:
                #   val = 6
                # if b'scissors' in pop and b'0' in pop:
                #   val = 7
                # if b'airplane' in pop and b'255' in pop:
                #   val = 8
                # if b'airplane' in pop and b'0' in pop:
                #   val = 9
                classesPop[0][i] = val
                population[clusterIDX, int(classesPop[0][i])] = classesPop[1][i] / sum(classesPop[1])
                probs[clusterIDX, int(classesPop[0][i])] = classesPop[1][i]/len(patches)
              # max_Pclass = classesPop[0][np.argmax(classesPop[1])]
              try:
                true_im = [p['ref'] for p in patches if
                           (p['idx'] in idxlist[0] and int(max_Pclass) == p['seg'])]
                ims_true.append(random.sample(true_im, k=5))
              except:
                print(max_Pclass)
              false_im = [p['ref'] for p in patches if (
                        p['idx'] in idxlist[0] and int(max_Pclass) != p['seg'])]
              try:
                ims_false.append(random.sample(false_im, k=5))
              except:
                ims_false.append(false_im)
              counter[0, clusterIDX] = np.sum(classesPop[1])
            # #       for i, pop in enumerate(classesPop[0]):
            # #         if b'elephant' in pop and b'255' in pop:
            # #           val = 4
            # #         if b'elephant' in pop and b'0' in pop:
            # #           val = 5
            # #         if b'dog' in pop and b'255' in pop:
            # #           val = 2
            # #         if b'dog' in pop and b'0' in pop:
            # #           val = 3
            # #         if b'fire hydrant' in pop and b'255' in pop:
            # #           val = 6
            # #         if b'fire hydrant' in pop and b'0' in pop:
            # #           val = 7
            # #         if b'train' in pop and b'255' in pop:
            # #           val = 8
            # #         if b'train' in pop and b'0' in pop:
            # #           val = 9
            # #         if b'airplane' in pop and b'255' in pop:
            # #           val = 0
            # #         if b'airplane' in pop and b'0' in pop:
            # #           val = 1
            # #         classesPop[0][i] = val
            # #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
            # #         counter[val] += 1
            MI1 = sklearn.metrics.mutual_info_score([p['seg'] for p in patches if p['idx'] in np.hstack(patches2Include)], [clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)])
            # MI2 = 0
            # for i in range(probs.shape[0]):
            #   for j in range(probs.shape[1]):
            #     if (probs[i, j] * sum(probs[:, j]) * sum(probs[i, :])) != 0:
            #       MI2 += probs[i, j] * np.log(probs[i, j] / (sum(probs[:, j]) * sum(probs[i, :])))

            # fig, ax = plt.subplots(figsize=(6, 20))
            # # counter = np.broadcast_to(counter.T, population.shape)
            # # population = np.divide(population, counter)
            # im = ax.imshow(population)
            # ax.set_yticks(np.arange(len(clustersNames)))
            # ax.set_xticks(np.arange(len(classesIdx)))
            # # for y in range(n):
            # #   for x in range(10):
            # #     label = population[x, y]
            # #     text_x = x
            # #     text_y = y
            # #     ax.text(text_x, text_y, label, color='black', ha='center', va='center')
            # cb = plt.colorbar(im)
            # # plt.show()
            # plt.savefig(
            #   'heatmaps_pascal/pureness/popHeatmap_{}_clusters_{}_eigvecs_temp.png'.format(n, vLen))
            # plt.close(fig)
            ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            # fig, axs = plt.subplots(n, 13)
            fig, axs = plt.subplots(n, 14, figsize=(20, n),
                                    gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 2.5, 5]})
            plt.subplots_adjust(hspace=0.4, wspace=0.2)
            # fig.suptitle('{} clusters, {} eigV - MI = {}'.format(n, vLen, MI1), fontsize=30)
            # plt.subplots_adjust(wspace=0.1, hspace=0.1, left=2, right=2.2)
            sortbyH = [sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0]) for i in range(n)]
            sortbyclass = [np.argmax(population[i, :]) for i in range(n)]
            sortedclustersIdx = np.lexsort((sortbyH, sortbyclass))
            # create similarity hists
            patchesBycluster = np.hstack(patches2Include)
            # distsFromClustersCenter = clusters.transform(Y)
            # distsperCluster = [distsFromClustersCenter[np.where(clusters.labels_ == i)] for i in range(n)]
            dis4Hist = np.zeros(dissim.shape)
            clustersSize = np.unique([clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)], return_counts=True)
            for i in range(len(patchesBycluster)):
              dis4Hist[i, :] = affinity[patchesBycluster[i], :]
            dissim_forcopy = copy.deepcopy(dis4Hist)
            for i in range(len(patchesBycluster)):
              dis4Hist[:, i] = dissim_forcopy[:, patchesBycluster[i]]
            dis4Hist = dis4Hist[:np.hstack(patches2Include).shape[0], :np.hstack(patches2Include).shape[0]]
            dists_within_cluster = []
            for i in range(n):
              if i == 0:
                dists_within_cluster.append(dis4Hist[:clustersSize[1][i], :clustersSize[1][i]])
              elif i == n-1:
                dists_within_cluster.append(dis4Hist[-clustersSize[1][i]:, -clustersSize[1][i]:])
              else:
                dists_within_cluster.append(dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                          sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)])])

            dists_between_cluster = []
            for i in range(n):
              if i == 0:
                dists_between_cluster.append(dis4Hist[:clustersSize[1][i], clustersSize[1][i]:])
              elif i == n-1:
                dists_between_cluster.append(dis4Hist[-clustersSize[1][i]:, :-clustersSize[1][i]])
              else:
                dists_between_cluster.append(np.hstack((dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                                      :sum(clustersSize[1][:i])],
                                                      dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                                      sum(clustersSize[1][:(i + 1)]):])))
            for k in range(n):
              i = sortedclustersIdx[k]
              idxlist = np.where(clusters.labels_ == i)
              #       print(len(idxlist[0])/len(patches))
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
              classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
              # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
              clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
              barvals = []
              for c in classesPopDS[0]:
                if c in classesPop[0]:
                  barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
                else:
                  barvals.append(0)
              ticks = range(len(classesPopDS[0]))
              # gs = axs[i, -2].get_gridspec()
              # remove the underlying axes
              # for ax in axs[i, 10:]:
              #   ax.remove()
              # axbig = fig.add_subplot(gs[i, 10:])
              axs[k, 0].set_ylabel(
                'cluster no.{} \n H={:.4f},\n dominant class:{} \n p_class={:.4f}'.format(i, clusterscondH, np.argmax(population[i, :]), np.max(population[i, :])),
                rotation=0, size='medium', loc='bottom')
              for j in range(14):
                if j == 0:
                  plt.setp(axs[k, j].spines.values(), color='white')
                  axs[k, j].set_xticks([])
                  axs[k, j].set_yticks([])
                elif j < 6:
                  axs[k, j].imshow(ims_true[i][j - 1] + ds_mean)
                  axs[k, j].set_xticks([])
                  axs[k, j].set_yticks([])
                  # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
                elif j == 6:
                  axs[k, j].remove()
                elif j < 12:
                  try:
                    axs[k, j].imshow(ims_false[i][j - 7] + ds_mean)
                    axs[k, j].set_xticks([])
                    axs[k, j].set_yticks([])
                    # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
                  except:
                    plt.setp(axs[k, j].spines.values(), color='white')
                    axs[k, j].set_xticks([])
                    axs[k, j].set_yticks([])
                elif j==12:
                  # ticks = range(len(classesPop[1]))
                  axs[k, j].bar(ticks, barvals, align='center')
                  axs[k, j].set_xticks(ticks)
                  axs[k, j].tick_params(axis='x', rotation=0)
                  # axs[i,j].hist(GT_cluster, )
                else:
                  axs[k, j].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
                  axs[k, j].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)

            fig.align_ylabels()
            # fig.tight_layout()
            plt.savefig(
              'heatmaps_pascal/LDA/samples/{}_clusters_{}_eigvecs_{:.4f}_MI.png'.format(n, vLen, MI1),
              bbox_inches='tight')
            plt.close(fig)
            # plt.show()
          # # simulate and test the population of one cluster
          #   population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
          #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
          #   clustersNames = range(n)
          #   labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background','elephant',
          #   'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
          #   for clusterIDX in range(n):
          #     idxlist = np.where(clusters.labels_ == clusterIDX)
          #     print(len(idxlist)/len(patches))
          #     GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
          #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
          #     for i, pop in enumerate(classesPop[0]):
          #       if b'airplane' in pop and b'255' in pop:
          #         classesPop[0][i] = 0
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'airplane' in pop and b'0' in pop:
          #         classesPop[0][i] = 1
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'dog' in pop and b'255' in pop:
          #         classesPop[0][i] = 2
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'dog' in pop and b'0' in pop:
          #         classesPop[0][i] = 3
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'fire hydrant' in pop and b'255' in pop:
          #         classesPop[0][i] = 6
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'fire hydrant' in pop and b'0' in pop:
          #         classesPop[0][i] = 7
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'elephant' in pop and b'255' in pop:
          #         classesPop[0][i] = 4
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'elephant' in pop and b'0' in pop:
          #         classesPop[0][i] = 5
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'train' in pop and b'255' in pop:
          #         classesPop[0][i] = 8
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'train' in pop and b'0' in pop:
          #         classesPop[0][i] = 9
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #   fig, ax = plt.subplots()
          #   im = ax.imshow(population)
          #   ax.set_yticks(np.arange(len(clustersNames)))
          #   ax.set_xticks(np.arange(len(labelsNames)))
          #   cb = plt.colorbar(im)
          #   plt.savefig('heatmaps_coco_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
          #   plt.close(fig)
            # population[clusterIDX, classesPop[0]] = classesPop[1]
            # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
            #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])

          # plt.show()
          # plt.savefig('keepDaway_newpatches.png')
          # plt.close(fig2)
          # fig, ax = plt.subplots()
          # ax.imshow(population)
          # ax.set_yticks(np.arange(len(clustersNames)), labels=clustersNames)
          # ax.set_xticks(np.arange(len(labelsNames)), labels=labelsNames)
        ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
        random.shuffle(ims)
        fig, axs = plt.subplots(6, 6)
        for i in range(6):
          for j in range(6):
            axs[i, j].imshow(ims[i * 6 + j])
        plt.show()
          # patches = sorted(patches, key=lambda p: p['idx'])
          # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
          # clustering_kmeans = SpectralClustering(n_clusters=50,
          #                                        assign_labels='kmeans',
          #                                        random_state=0,
          #                                        affinity='nearest_neighbors').fit(
          #   np.array([p['patch'].flatten() for p in patches]))
          # clustering_disc = SpectralClustering(n_clusters=50,
          #                                      assign_labels='discretize',
          #                                      random_state=0,
          #                                      affinity='nearest_neighbors').fit(
          #   np.array([p['patch'].flatten() for p in patches]))
          #
          # plt.hist(clustering_disc.labels_, bins=50)
          # plt.hist(clustering_kmeans.labels_, bins=50)


      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  elif MIT:
    image, label, imname = getRep_MIT()

    _, _, fuse = resnet50.inference(image, tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]), tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1)) as sess:
      sess.run(tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables()))
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
    # #
    # # _, _, fuse = resnet50.inference(image, tf.zeros(
    # #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    # #    resnet50_input.NUM_CLASSES]), tf.zeros(
    # #   [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
    # #    resnet50_input.NUM_CLASSES]))
    # # # ### bsds model
    # # saver = tf.train.Saver(tf.global_variables())
    # # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
    # #                                       intra_op_parallelism_threads=1)) as sess:
    # #   sess.run(tf.group(tf.initialize_all_variables(),
    # #                     tf.initialize_local_variables()))
    # #   ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # #   if ckpt and ckpt.model_checkpoint_path:
    # #     saver.restore(sess, ckpt.model_checkpoint_path)
    # #   else:
    # #     print('No checkpoint file found')
    # #     return
    # # # ### pascal weights
    # # # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
    # # #                                       intra_op_parallelism_threads=1)) as sess:
    # # #   model_dict = np.load('fp_weights.npy', allow_pickle=True).item()
    # # #   all_vars = tf.trainable_variables()
    # # #   for v in all_vars:
    # # #     if (v.op.name.find("weights") > -1) and (v.op.name.find("softmax_linear") == -1) and (
    # # #             v.op.name.find("fuse") == -1) and (v.op.name.find("conv5_3") == -1):
    # # #       assign_op = v.assign(model_dict[v.op.name])
    # # #       sess.run(assign_op)
    # # #
    # #   sess.run(tf.group(tf.initialize_all_variables(),
    # #                     tf.initialize_local_variables()))
    # # #   # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # # #   # if ckpt and ckpt.model_checkpoint_path:
    # # #   #   saver.restore(sess, ckpt.model_checkpoint_path)
    # # #   # else:
    # # #   #   print('No checkpoint file found')
    # # #   #   return
    # # #   # Start the queue runners.
    # #   coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
                                           start=True))
        # patches = getPatchDS_MIT(sess, image, label, imname, fuse, patch_size=8, ifPCA=True, n=16)
        # # patches = getPatchDS_frompkl()
        #
        # with open('patches_8_MIT_pca16_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(patches, f)
        with open('patches_8_MIT_pca16_2024-04-27 21:14:05.394220.pkl', 'rb') as f:
          patches = pickle.load(f)
        # with open('patches_8_MIT_pca16_balanced_noBG_2024-03-07 12:22:39.858088.pkl', 'rb') as f:
        #   patches = pickle.load(f)
        # classes2inc = [82, 99, 123]
        # patches = [p for p in patches if p['seg'] in classes2inc]
        # for i in range(len(patches)):
        #   patches[i]['idx']=i
        # new_patches = random.sample(patches, 1800)
        # with open('patches_8_pascal_pca16_filtered_sampled.pkl', 'rb') as f:
        #   new_patches = pickle.load(f)
        # lda = LDA(new_patches)
  #     # dissim =
  # #
  # # # # #       # patches = random.choices(patches, k=500)
  # # # # #       # threads_per_block = (16,16)
  # # # # #       # blocks_per_grid = ((4224+16-1)//16, (4224+16-1)//16)
  # # # # #       # patches_array = [p['patch'] for p in patches]
  # # # # #       # patches_cuda = cuda.to_device(patches_array)
  #       doues = comb(new_patches, 2)
  # #       # doues = [(d, lda) for d in doues]
  # # # # # # # # #       # dissim = calcPatchDistmulti(doues)
  #       data = []
  #       labels = []
  # #       # p1order = []
  # #       # p2order = []
  #       with multiprocessing.Pool(processes=16) as p:
  #           with tqdm(total=(len(new_patches) ** 2)/2) as pbar:
  #             for out in p.imap_unordered(calcPatchDist4lda, doues):
  #               pbar.update()
  #               _, _, d, l = out
  #               # p1 = np.where([p['idx']==p1 for p in new_patches])
  #               # p2 = np.where([p['idx']==p2 for p in new_patches])
  #               # p1order.append(p1)
  #               # p2order.append(p2)
  #               data.append(d)
  #               labels.append(int(l))
  #       data = np.array(data)
  #       labels = np.array(labels)
  #       weights = lda_fromScratch(data, labels)
  #       patches = [p for p in patches if p['seg']!=np.uint8(30)]
  #       doues = comb(patches, 2)
  #       dissim = np.zeros((len(patches), len(patches)))
        # dists = []
  #       with multiprocessing.Pool(processes=16) as p:
  #         with tqdm(total=(len(patches) ** 2) / 2) as pbar:
  #           for d in p.imap_unordered(calcPatchDist4lda, doues):
  #             pbar.update()
  #             p1, p2, dist, l = d
              # dissim[p1,p2] = dissim[p2,p1] = np.array(dist).dot(weights)
  #       # with multiprocessing.Pool(processes=16) as p:
  #       #     with tqdm(total=(len(patches) ** 2)/2) as pbar:
  #       #       for d in p.imap_unordered(transorm2LDA, doues):
  #       #         pbar.update()
  #       #         dissim[d[0], d[1]] = d[2]
  #       #         dissim[d[1], d[0]] = d[2]
  # # # # #       # dissim = cuda.to_device(dissim)
  # # # # #       # dissim = calcDistGPUPar[blocks_per_grid,threads_per_block](patches_cuda, dissim)

        # with multiprocessing.Pool(processes=16) as p:
        #   with tqdm(total=(len(patches) ** 2)/2) as pbar:
        #     for d in p.imap_unordered(calcPatchDistmulti, doues):
        #       pbar.update()
        #       dissim[d[0], d[1]] = d[2]
        #       dissim[d[1], d[0]] = d[2]
              # dists.append([d[3], (d[0], d[1])])
  # # # #       # # for p1 in tqdm(patches):
  # # # #       # #   for p2 in patches:
  # # # #       # #     if dissim[p1['idx'], p2['idx']] != 0:
  # # # #       # #       continue
  # # # #       # #     else:
  # # # #       # #       # start = time.time()
  # # # #       # #       # patch1 = torch.from_numpy(p1['patch'])
  # # # #       # #       # patch2 = torch.from_numpy(p2['patch'])
  # # # #       # #       # dist = calcPatchDisttorch(patch1, patch2)
  # # # #       # #       dist = calcPatchDist(p1['patch'], p2['patch'])
  # # # #       # #       dissim[p1['idx'], p2['idx']] = dist
  # # # #       # #       dissim[p2['idx'], p1['idx']] = dist
  # # # #       # #       # print(time.time()-start)
  #       dissim = dissim[np.any(dissim != 0, axis=1), :][:, np.any(dissim != 0, axis=1)]
  #       with open('dissim_8_MIT_{}.pkl'.format(datetime.now()), 'wb') as f:
  #          pickle.dump(dissim, f)

        with open('dissim_8_MIT_2024-04-28 09:17:28.297780.pkl', 'rb') as f:
          dissim = pickle.load(f)
  #      with open('dissim_8_pascal_avg_filtered_q0_noBG_balanced_2024-03-07 13:00:44.615576.pkl', 'rb') as f:
  #        dissim = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q50_2023-09-28 14:59:31.762212.pkl', 'rb') as f:
  #         dissim_q50 = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q80_2023-09-27 16:45:36.574020.pkl', 'rb') as f:
  #         dissim_q80 = pickle.load(f)
  #       with open('dissim_8_pascal_avg_filtered_q100_2023-09-27 17:38:53.305407.pkl', 'rb') as f:
  #         dissim_q100 = pickle.load(f)

        max_d = np.amax(dissim)
        # affinity = 1 - (dissim / max_d)
        sigma = np.percentile(dissim, 1)
        # dissim[dissim>np.percentile(dissim, 10)]=np.amax(dissim)
        affinity = np.exp(-dissim/sigma)
        # affinity[affinity<np.percentile(affinity, 95)]=0
        D = np.zeros(affinity.shape)
        for i in range(affinity.shape[0]):
          D[i, i] = sum(affinity[i, :])
        L = D- affinity
        L_sym = (np.diag(np.diag(D) ** (-1 / 2)) @ L) @ np.diag(np.diag(D) ** (-1 / 2))
        eig = scipy.linalg.eig(L_sym)
        idx = eig[0].argsort()
        eigenValues = eig[0][idx]
        eigenVectors = eig[1][:, idx]
        # with open('eig_balanced_MIT_pca16_{}.pkl'.format(datetime.now()), 'wb') as f:
        #   pickle.dump(eig, f)
        # vLens = range(10, 151, 10)
        vLens = [20]
        clustersCenterD = []
        inClustersD = []
        ns = range(20,401,20)
        # ns = [30]
        # fig2, ax2 = plt.subplots()
        segvals = [(p['seg']) for p in patches]
        classesPopDS = np.unique(segvals, return_counts=True, axis=0)
        classesIdx = {i: bytes(classesPopDS[0][i]) for i in range(classesPopDS[0].shape[0])}
        MIs = {}
        conHs = {}
        condHperC = {}
        purity = np.arange(0.1, 1.1, 0.1)
        chance2cl = {}
        coverage = {}
        for n in tqdm(ns):
          MI_n = []
        #   ims_true = []
        #   ims_false = []
        #   space = eig[1][:, :n]
        #   normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
        #   Y = (space.T / normalize_f).T
        #
        #   Kmeans = sklearn.cluster.KMeans(n_clusters=n, n_init=500)
        #   clusters = Kmeans.fit(Y)
        #   distsFromClustersCenter = clusters.transform(Y)
        #   distsperCluster = [
        #     (distsFromClustersCenter[np.where(clusters.labels_ == i)], np.where(clusters.labels_ == i)) for i in
        #     range(n)]
        #   patches2Include = [distsperCluster[i][1][0][np.where(
        #     distsperCluster[i][0][:, i] <= np.percentile(distsperCluster[i][0][:, i], 100))[0]] for i in range(n)]
        #   # clusters_trans = Kmeans.fit_transform(Y)
        #   # clustersCenterD.append(np.mean(
        #   #   [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
        #   #    for j in range(n) if i != j]))
        #   # inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
        # # # plot the change in the distances according to the size of the new patches descriptors
        # #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
        # #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
        # #   ax2.set_xlabel('# eig vec')
        # #   ax2.set_ylabel('mean distance')
        # #   ax2.legend()
        #   # simulate and test the population of one cluster
        #   population = np.zeros((n, len(classesPopDS[0])))
        #   probs = np.zeros(population.shape)
        #   # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
        #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
        #   clustersNames = range(n)
        #   # labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
        #   #                'scissors background', 'airplane', 'airplane background']
        #   # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
        #   #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
        #   counter = np.zeros((1, n))
        #   for clusterIDX in range(n):
        #     # idxlist = np.where(clusters.labels_ == clusterIDX)
        #     idxlist = [patches2Include[clusterIDX]]
        #     #       print(len(idxlist[0])/len(patches))
        #     GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
        #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
        #     max_Pclass = copy.deepcopy(classesPop[0])[np.argmax(classesPop[1])]
        #     for i, pop in enumerate(classesPop[0]):
        #       val = list(classesIdx.values()).index(pop)
        #       # population[clusterIDX, val] = classesPop[1][i] / len(idxlist[0])
        #       # if b'bus' in pop and b'255' in pop:
        #       #   val = 0
        #       # if b'bus' in pop and b'0' in pop:
        #       #   val = 1
        #       # if b'dog' in pop and b'255' in pop:
        #       #   val = 2
        #       # if b'dog' in pop and b'0' in pop:
        #       #   val = 3
        #       # if b'pizza' in pop and b'255' in pop:
        #       #   val = 4
        #       # if b'pizza' in pop and b'0' in pop:
        #       #   val = 5
        #       # if b'scissors' in pop and b'255' in pop:
        #       #   val = 6
        #       # if b'scissors' in pop and b'0' in pop:
        #       #   val = 7
        #       # if b'airplane' in pop and b'255' in pop:
        #       #   val = 8
        #       # if b'airplane' in pop and b'0' in pop:
        #       #   val = 9
        #       classesPop[0][i] = val
        #       population[clusterIDX, int(classesPop[0][i])] = classesPop[1][i] / sum(classesPop[1])
        #       probs[clusterIDX, int(classesPop[0][i])] = classesPop[1][i]/len(patches)
        #     # max_Pclass = classesPop[0][np.argmax(classesPop[1])]
        #     try:
        #       true_im = [p['ref'] for p in patches if
        #                  (p['idx'] in idxlist[0] and int(max_Pclass) == p['seg'])]
        #       ims_true.append(random.sample(true_im, k=5))
        #     except:
        #       print(max_Pclass)
        #     false_im = [p['ref'] for p in patches if (
        #               p['idx'] in idxlist[0] and int(max_Pclass) != p['seg'])]
        #     try:
        #       ims_false.append(random.sample(false_im, k=5))
        #     except:
        #       ims_false.append(false_im)
        #     counter[0, clusterIDX] = np.sum(classesPop[1])
        #   # #       for i, pop in enumerate(classesPop[0]):
        #   # #         if b'elephant' in pop and b'255' in pop:
        #   # #           val = 4
        #   # #         if b'elephant' in pop and b'0' in pop:
        #   # #           val = 5
        #   # #         if b'dog' in pop and b'255' in pop:
        #   # #           val = 2
        #   # #         if b'dog' in pop and b'0' in pop:
        #   # #           val = 3
        #   # #         if b'fire hydrant' in pop and b'255' in pop:
        #   # #           val = 6
        #   # #         if b'fire hydrant' in pop and b'0' in pop:
        #   # #           val = 7
        #   # #         if b'train' in pop and b'255' in pop:
        #   # #           val = 8
        #   # #         if b'train' in pop and b'0' in pop:
        #   # #           val = 9
        #   # #         if b'airplane' in pop and b'255' in pop:
        #   # #           val = 0
        #   # #         if b'airplane' in pop and b'0' in pop:
        #   # #           val = 1
        #   # #         classesPop[0][i] = val
        #   # #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
        #   # #         counter[val] += 1
        #   MI1 = sklearn.metrics.mutual_info_score([p['seg'] for p in patches if p['idx'] in np.hstack(patches2Include)], [clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)])
        #   MI_n.append(MI1)
        #   # MI2 = 0
        #   # for i in range(probs.shape[0]):
        #   #   for j in range(probs.shape[1]):
        #   #     if (probs[i, j] * sum(probs[:, j]) * sum(probs[i, :])) != 0:
        #   #       MI2 += probs[i, j] * np.log(probs[i, j] / (sum(probs[:, j]) * sum(probs[i, :])))
        #
        #   # fig, ax = plt.subplots(figsize=(6, 20))
        #   # # counter = np.broadcast_to(counter.T, population.shape)
        #   # # population = np.divide(population, counter)
        #   # im = ax.imshow(population)
        #   # ax.set_yticks(np.arange(len(clustersNames)))
        #   # ax.set_xticks(np.arange(len(classesIdx)))
        #   # # for y in range(n):
        #   # #   for x in range(10):
        #   # #     label = population[x, y]
        #   # #     text_x = x
        #   # #     text_y = y
        #   # #     ax.text(text_x, text_y, label, color='black', ha='center', va='center')
        #   # cb = plt.colorbar(im)
        #   # # plt.show()
        #   # plt.savefig(
        #   #   'heatmaps_pascal/pureness/popHeatmap_{}_clusters_{}_eigvecs_temp.png'.format(n, vLen))
        #   # plt.close(fig)
        #   # ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        #   # fig, axs = plt.subplots(n, 13)
        #   # fig, axs = plt.subplots(n, 14, figsize=(20, n),
        #   #                         gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 2.5, 5]})
        #   # plt.subplots_adjust(hspace=0.4, wspace=0.2)
        #   # fig.suptitle('{} clusters, {} eigV - MI = {}'.format(n, vLen, MI1), fontsize=30)
        #   # plt.subplots_adjust(wspace=0.1, hspace=0.1, left=2, right=2.2)
        #   sortbyH = [sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0]) for i in range(n)]
        #   sortbyclass = [np.argmax(population[i, :]) for i in range(n)]
        #   sortedclustersIdx = np.lexsort((sortbyH, sortbyclass))
        #   # create similarity hists
        #   patchesBycluster = np.hstack(patches2Include)
        #   # distsFromClustersCenter = clusters.transform(Y)
        #   # distsperCluster = [distsFromClustersCenter[np.where(clusters.labels_ == i)] for i in range(n)]
        #   dis4Hist = np.zeros(dissim.shape)
        #   clustersSize = np.unique([clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)], return_counts=True)
        #   for i in range(len(patchesBycluster)):
        #     dis4Hist[i, :] = affinity[patchesBycluster[i], :]
        #   dissim_forcopy = copy.deepcopy(dis4Hist)
        #   for i in range(len(patchesBycluster)):
        #     dis4Hist[:, i] = dissim_forcopy[:, patchesBycluster[i]]
        #   dis4Hist = dis4Hist[:np.hstack(patches2Include).shape[0], :np.hstack(patches2Include).shape[0]]
        #   dists_within_cluster = []
        #   for i in range(n):
        #     if i == 0:
        #       dists_within_cluster.append(dis4Hist[:clustersSize[1][i], :clustersSize[1][i]])
        #     elif i == n-1:
        #       dists_within_cluster.append(dis4Hist[-clustersSize[1][i]:, -clustersSize[1][i]:])
        #     else:
        #       dists_within_cluster.append(dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                 sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)])])
        #
        #   dists_between_cluster = []
        #   for i in range(n):
        #     if i == 0:
        #       dists_between_cluster.append(dis4Hist[:clustersSize[1][i], clustersSize[1][i]:])
        #     elif i == n-1:
        #       dists_between_cluster.append(dis4Hist[-clustersSize[1][i]:, :-clustersSize[1][i]])
        #     else:
        #       dists_between_cluster.append(np.hstack((dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                             :sum(clustersSize[1][:i])],
        #                                             dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
        #                                             sum(clustersSize[1][:(i + 1)]):])))
        #   condH = []
        #   for k in range(n):
        #     i = sortedclustersIdx[k]
        #     idxlist = np.where(clusters.labels_ == i)
        #     #       print(len(idxlist[0])/len(patches))
        #     GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
        #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
        #     # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
        #     clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
        #     condH.append(clusterscondH)
        #   #   barvals = []
        #   #   for c in classesPopDS[0]:
        #   #     if c in classesPop[0]:
        #   #       barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
        #   #     else:
        #   #       barvals.append(0)
        #   #   ticks = range(18)
        #   #   # gs = axs[i, -2].get_gridspec()
        #   #   # remove the underlying axes
        #   #   # for ax in axs[i, 10:]:
        #   #   #   ax.remove()
        #   #   # axbig = fig.add_subplot(gs[i, 10:])
        #   #   axs[k, 0].set_ylabel(
        #   #     'cluster no.{} \n H={:.4f},\n dominant class:{} \n p_class={:.4f}'.format(i, clusterscondH, np.argmax(population[i, :]), np.max(population[i, :])),
        #   #     rotation=0, size='medium', loc='bottom')
        #   #   for j in range(14):
        #   #     if j == 0:
        #   #       plt.setp(axs[k, j].spines.values(), color='white')
        #   #       axs[k, j].set_xticks([])
        #   #       axs[k, j].set_yticks([])
        #   #     elif j < 6:
        #   #       axs[k, j].imshow(ims_true[i][j - 1])
        #   #       axs[k, j].set_xticks([])
        #   #       axs[k, j].set_yticks([])
        #   #       # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
        #   #     elif j == 6:
        #   #       axs[k, j].remove()
        #   #     elif j < 12:
        #   #       try:
        #   #         axs[k, j].imshow(ims_false[i][j - 7])
        #   #         axs[k, j].set_xticks([])
        #   #         axs[k, j].set_yticks([])
        #   #         # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
        #   #       except:
        #   #         plt.setp(axs[k, j].spines.values(), color='white')
        #   #         axs[k, j].set_xticks([])
        #   #         axs[k, j].set_yticks([])
        #   #     elif j==12:
        #   #       # ticks = range(len(classesPop[1]))
        #   #       axs[k, j].bar(ticks, barvals, align='center')
        #   #       axs[k, j].set_xticks(ticks)
        #   #       axs[k, j].tick_params(axis='x', rotation=0)
        #   #       # axs[i,j].hist(GT_cluster, )
        #   #     else:
        #   #       axs[k, j].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
        #   #       axs[k, j].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)
        #   #
        #   # fig.align_ylabels()
        #   # # fig.tight_layout()
        #   # plt.savefig(
        #   #   'Thesis figures/pascal/clustering/noBG/{}_clusters_{}_eigvecs_{:.4f}_MI.png'.format(n, vLen, MI1),
        #   #   bbox_inches='tight')
        #   # plt.close(fig)
        #
        #   conHs[str(n) + ' ' + str(n)] = condH
        #   condHperClass = {}
        #   for m in range(len(classesPopDS[0])):
        #     Hs = []
        #     for i, c in enumerate(patches2Include):
        #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i,:])==np.flatnonzero(list(classesIdx.values())==p['seg']))]
        #       # if np.max(population[i,:]) >= purity:
        #       #   chance2classify[np.argmax(population[i,:])] += len(ps)/classesPopDS[1][np.argmax(population[i,:])]
        #       for p in ps:
        #         # print(type(classesIdx[m]))
        #         if p['seg'] == classesIdx[m]:
        #           Hs.append(condH[i])
        #     condHperClass[m] = Hs
        #   for pu in purity:
        #     chance2classify = np.zeros(len(classesPopDS[0]))
        #     for i, c in enumerate(patches2Include):
        #       ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i, :]) == np.flatnonzero(
        #         list(classesIdx.values()) == p['seg']))]
        #       if np.max(population[i, :]) >= pu:
        #         chance2classify[np.argmax(population[i, :])] += len(ps) / classesPopDS[1][np.argmax(population[i, :])]
        #     chance2cl[str(n)+'_'+str(n)+'_'+str(pu)] = chance2classify
        #   condHperC[str(n)+'_'+str(n)] = condHperClass
          for vLen in vLens:
            vLen=n
            ims_true = []
            ims_false = []
            space = eigenVectors[:, :vLen]
            normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
            Y = (space.T / normalize_f).T
            Kmeans = sklearn.cluster.KMeans(n_clusters=n, n_init=50)
            clusters = Kmeans.fit(Y)
            distsFromClustersCenter = clusters.transform(Y)
            distsperCluster = [
              (distsFromClustersCenter[np.where(clusters.labels_ == i)], np.where(clusters.labels_ == i)) for i in
              range(n)]
            patches2Include = [distsperCluster[i][1][0][np.where(
              distsperCluster[i][0][:, i] <= np.percentile(distsperCluster[i][0][:, i], 100))[0]] for i in range(n)]
            # clusters_trans = Kmeans.fit_transform(Y)
            # clustersCenterD.append(np.mean(
            #   [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
            #    for j in range(n) if i != j]))
            # inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
          # # plot the change in the distances according to the size of the new patches descriptors
          #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='between cluster mean distance n={}'.format(n))
          #   ax2.plot(vLens, inClustersD[-len(vLens):], label='within clusters mean distance n={}'.format(n))
          #   ax2.set_xlabel('# eig vec')
          #   ax2.set_ylabel('mean distance')
          #   ax2.legend()
            # simulate and test the population of one cluster
            population = np.zeros((n, len(classesPopDS[0])))
            probs = np.zeros(population.shape)
            # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
            # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
            clustersNames = range(n)
            # labelsNames = ['bus', 'bus backgrouns', 'dog', 'dog background', 'pizza', 'pizza background', 'scissors',
            #                'scissors background', 'airplane', 'airplane background']
            # labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background', 'elephant',
            #                'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
            counter = np.zeros((1, n))
            for clusterIDX in range(n):
              # idxlist = np.where(clusters.labels_ == clusterIDX)
              idxlist = [patches2Include[clusterIDX]]
              #       print(len(idxlist[0])/len(patches))
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
              classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
              max_Pclass = copy.deepcopy(classesPop[0])[np.argmax(classesPop[1])]
              for i, pop in enumerate(classesPop[0]):
                val = list(classesIdx.values()).index(pop)
                # population[clusterIDX, val] = classesPop[1][i] / len(idxlist[0])
                # if b'bus' in pop and b'255' in pop:
                #   val = 0
                # if b'bus' in pop and b'0' in pop:
                #   val = 1
                # if b'dog' in pop and b'255' in pop:
                #   val = 2
                # if b'dog' in pop and b'0' in pop:
                #   val = 3
                # if b'pizza' in pop and b'255' in pop:
                #   val = 4
                # if b'pizza' in pop and b'0' in pop:
                #   val = 5
                # if b'scissors' in pop and b'255' in pop:
                #   val = 6
                # if b'scissors' in pop and b'0' in pop:
                #   val = 7
                # if b'airplane' in pop and b'255' in pop:
                #   val = 8
                # if b'airplane' in pop and b'0' in pop:
                #   val = 9
                classesPop[0][i] = val
                population[clusterIDX, int(classesPop[0][i])] = classesPop[1][i] / sum(classesPop[1])
                probs[clusterIDX, int(classesPop[0][i])] = classesPop[1][i]/len(patches)
              # max_Pclass = classesPop[0][np.argmax(classesPop[1])]
              try:
                true_im = [p['ref'] for p in patches if
                           (p['idx'] in idxlist[0] and max_Pclass == p['seg'])]
                ims_true.append(random.sample(true_im, k=5))
              except:
                print(max_Pclass)
              false_im = [p['ref'] for p in patches if (
                        p['idx'] in idxlist[0] and max_Pclass != p['seg'])]
              try:
                ims_false.append(random.sample(false_im, k=5))
              except:
                ims_false.append(false_im)
              counter[0, clusterIDX] = np.sum(classesPop[1])
            # #       for i, pop in enumerate(classesPop[0]):
            # #         if b'elephant' in pop and b'255' in pop:
            # #           val = 4
            # #         if b'elephant' in pop and b'0' in pop:
            # #           val = 5
            # #         if b'dog' in pop and b'255' in pop:
            # #           val = 2
            # #         if b'dog' in pop and b'0' in pop:
            # #           val = 3
            # #         if b'fire hydrant' in pop and b'255' in pop:
            # #           val = 6
            # #         if b'fire hydrant' in pop and b'0' in pop:
            # #           val = 7
            # #         if b'train' in pop and b'255' in pop:
            # #           val = 8
            # #         if b'train' in pop and b'0' in pop:
            # #           val = 9
            # #         if b'airplane' in pop and b'255' in pop:
            # #           val = 0
            # #         if b'airplane' in pop and b'0' in pop:
            # #           val = 1
            # #         classesPop[0][i] = val
            # #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
            # #         counter[val] += 1
            MI1 = sklearn.metrics.mutual_info_score([p['seg'] for p in patches if p['idx'] in np.hstack(patches2Include)], [clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)])
            MI_n.append(MI1)
            # MI2 = 0
            # for i in range(probs.shape[0]):
            #   for j in range(probs.shape[1]):
            #     if (probs[i, j] * sum(probs[:, j]) * sum(probs[i, :])) != 0:
            #       MI2 += probs[i, j] * np.log(probs[i, j] / (sum(probs[:, j]) * sum(probs[i, :])))

            # fig, ax = plt.subplots(figsize=(6, 20))
            # # counter = np.broadcast_to(counter.T, population.shape)
            # # population = np.divide(population, counter)
            # im = ax.imshow(population)
            # ax.set_yticks(np.arange(len(clustersNames)))
            # ax.set_xticks(np.arange(len(classesIdx)))
            # # for y in range(n):
            # #   for x in range(10):
            # #     label = population[x, y]
            # #     text_x = x
            # #     text_y = y
            # #     ax.text(text_x, text_y, label, color='black', ha='center', va='center')
            # cb = plt.colorbar(im)
            # # plt.show()
            # plt.savefig(
            #   'heatmaps_pascal/pureness/popHeatmap_{}_clusters_{}_eigvecs_temp.png'.format(n, vLen))
            # plt.close(fig)
            # ds_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            # fig, axs = plt.subplots(n, 13)
            fig, axs = plt.subplots(19, 14, figsize=(15, 20),
                                    gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 2.5, 5]})
            plt.subplots_adjust(hspace=0.4, wspace=0.2)
            fig.suptitle('{} clusters, MI = {}'.format(n, MI1))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)#, left=2, right=2.2)
            sortbyH = [sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0]) for i in range(n)]
            sortbyclass = [np.argmax(population[i, :]) for i in range(n)]
            sortedclustersIdx = np.lexsort((sortbyH, sortbyclass))
            # create similarity hists
            patchesBycluster = np.hstack(patches2Include)
            # distsFromClustersCenter = clusters.transform(Y)
            # distsperCluster = [distsFromClustersCenter[np.where(clusters.labels_ == i)] for i in range(n)]
            dis4Hist = np.zeros(dissim.shape)
            clustersSize = np.unique([clusters.labels_[i] for i in range(len(clusters.labels_)) if i in np.hstack(patches2Include)], return_counts=True)
            for i in range(len(patchesBycluster)):
              dis4Hist[i, :] = affinity[patchesBycluster[i], :]
            dissim_forcopy = copy.deepcopy(dis4Hist)
            for i in range(len(patchesBycluster)):
              dis4Hist[:, i] = dissim_forcopy[:, patchesBycluster[i]]
            dis4Hist = dis4Hist[:np.hstack(patches2Include).shape[0], :np.hstack(patches2Include).shape[0]]
            dists_within_cluster = []
            for i in range(n):
              if i == 0:
                dists_within_cluster.append(dis4Hist[:clustersSize[1][i], :clustersSize[1][i]])
              elif i == n-1:
                dists_within_cluster.append(dis4Hist[-clustersSize[1][i]:, -clustersSize[1][i]:])
              else:
                dists_within_cluster.append(dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                          sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)])])

            dists_between_cluster = []
            for i in range(n):
              if i == 0:
                dists_between_cluster.append(dis4Hist[:clustersSize[1][i], clustersSize[1][i]:])
              elif i == n-1:
                dists_between_cluster.append(dis4Hist[-clustersSize[1][i]:, :-clustersSize[1][i]])
              else:
                dists_between_cluster.append(np.hstack((dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                                      :sum(clustersSize[1][:i])],
                                                      dis4Hist[sum(clustersSize[1][:i]):sum(clustersSize[1][:(i + 1)]),
                                                      sum(clustersSize[1][:(i + 1)]):])))
            clusters2show = {}
            condH = []
            for k in range(n):
              ind = 0
              if clusters2show.values():
                ind = sum(clusters2show.values())
              i = sortedclustersIdx[k]

              idxlist = np.where(clusters.labels_ == i)
              #       print(len(idxlist[0])/len(patches))
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
              classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
              # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
              clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
              condH.append(clusterscondH)
              if str(np.argmax(population[i, :])) in clusters2show.keys():
                if clusters2show[str(np.argmax(population[i, :]))]==2:
                  continue
                else:
                  clusters2show[str(np.argmax(population[i, :]))] = 2
              else:
                clusters2show[str(np.argmax(population[i, :]))]=1
              barvals = []
              for c in classesPopDS[0]:
                if c in classesPop[0]:
                  barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
                else:
                  barvals.append(0)
            #   ticks = range(18)
            #   # gs = axs[i, -2].get_gridspec()
            #   # remove the underlying axes
            #   # for ax in axs[i, 10:]:
            #   #   ax.remove()
            #   # axbig = fig.add_subplot(gs[i, 10:])
              axs[ind, 0].set_ylabel(
                'cluster no.{} \n H={:.4f},\n dominant class:{} \n p_class={:.4f}'.format(i, clusterscondH, np.argmax(population[i, :]), np.max(population[i, :])),
                rotation=0, size='medium', loc='bottom')
              for j in range(14):
                if j == 0:
                  plt.setp(axs[ind, j].spines.values(), color='white')
                  axs[ind, j].set_xticks([])
                  axs[ind, j].set_yticks([])
                elif j < 6:
                  axs[ind, j].imshow(ims_true[i][j - 1])
                  axs[ind, j].set_xticks([])
                  axs[ind, j].set_yticks([])
                  # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
                elif j == 6:
                  axs[ind, j].remove()
                elif j < 12:
                  try:
                    axs[ind, j].imshow(ims_false[i][j - 7])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                    # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                elif j==12:
                  ticks = range(len(barvals))
                  axs[ind, j].bar(ticks, barvals, align='center')
                  axs[ind, j].set_xticks(ticks)
                  axs[ind, j].tick_params(axis='x', rotation=0)
                  # axs[i,j].hist(GT_cluster, )
                else:
                  axs[ind, j].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
                  axs[ind, j].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)

            for ind, k in enumerate(np.argsort(sortbyH)[-5:]):

              i = k
              ind += sum(clusters2show.values())
              idxlist = np.where(clusters.labels_ == i)
              #       print(len(idxlist[0])/len(patches))
              GT_cluster = [(p['seg']) for p in patches if p['idx'] in idxlist[0]]
              classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
              # ticks = [classesPop[0][i][0]+' '+classesPop[0][i][1] for i in range(len(classesPop[0]))]
              clusterscondH = sum([(probs[i, j]/sum(probs[i,:])) * np.log(1/(probs[i, j]/sum(probs[i,:]))) for j in range(len(classesPopDS[0])) if probs[i, j] != 0])
              condH.append(clusterscondH)
              barvals = []
              for c in classesPopDS[0]:
                if c in classesPop[0]:
                  barvals.append(classesPop[1][np.where(classesPop[0] == c)][0].item())
                else:
                  barvals.append(0)
            #   ticks = range(18)
            #   # gs = axs[i, -2].get_gridspec()
            #   # remove the underlying axes
            #   # for ax in axs[i, 10:]:
            #   #   ax.remove()
            #   # axbig = fig.add_subplot(gs[i, 10:])
              axs[ind, 0].set_ylabel(
                'cluster no.{} \n H={:.4f},\n dominant class:{} \n p_class={:.4f}'.format(i, clusterscondH, np.argmax(population[i, :]), np.max(population[i, :])),
                rotation=0, size='medium', loc='bottom')
              for j in range(14):
                if j == 0:
                  plt.setp(axs[ind, j].spines.values(), color='white')
                  axs[ind, j].set_xticks([])
                  axs[ind, j].set_yticks([])
                elif j < 6:
                  axs[ind, j].imshow(ims_true[i][j - 1])
                  axs[ind, j].set_xticks([])
                  axs[ind, j].set_yticks([])
                  # plt.setp(axs[i, j].spines.values(),color='green', linewidth=5)
                elif j == 6:
                  axs[ind, j].remove()
                elif j < 12:
                  try:
                    axs[ind, j].imshow(ims_false[i][j - 7])
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                    # plt.setp(axs[i, j].spines.values(), color='red', linewidth=5)
                  except:
                    plt.setp(axs[ind, j].spines.values(), color='white')
                    axs[ind, j].set_xticks([])
                    axs[ind, j].set_yticks([])
                elif j==12:
                  ticks = range(len(barvals))
                  axs[ind, j].bar(ticks, barvals, align='center')
                  axs[ind, j].set_xticks(ticks)
                  axs[ind, j].tick_params(axis='x', rotation=0)
                  # axs[i,j].hist(GT_cluster, )
                else:
                  axs[ind, j].hist(dists_within_cluster[k][dists_within_cluster[k]!=1].flatten(), density=True, bins=50)
                  axs[ind, j].hist(dists_between_cluster[k].flatten(), density=True, bins=50, alpha=0.5)
            fig.align_ylabels()
            # fig.tight_layout()
            plt.savefig(
              'Thesis figures/MIT/samples/filtered_{}_clusters_{:.4f}_MI.png'.format(n, MI1),
              bbox_inches='tight')
            # plt.show()
            plt.close(fig)

            conHs[str(n) + ' ' + str(vLen)] = condH
            condHperClass = {}
            for m in range(len(classesPopDS[0])):
              Hs = []
              for i, c in enumerate(patches2Include):
                ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i,:])==list(classesIdx.values()).index(p['seg']))]
                # if np.max(population[i,:]) >= purity:
                #   chance2classify[np.argmax(population[i,:])] += len(ps)/classesPopDS[1][np.argmax(population[i,:])]
                for p in ps:
                  # print(type(classesIdx[m]))
                  if p['seg'] == classesIdx[m]:
                    Hs.append(condH[i])
              condHperClass[m] = Hs
            for pu in purity:
              chance2classify = np.zeros(len(classesPopDS[0]))
              cov = np.zeros(len(classesPopDS[0]))
              for i, c in enumerate(patches2Include):
                ps = [p for p in patches if (p['idx'] in c and np.argmax(population[i, :]) == list(classesIdx.values()).index(p['seg']))]
                if np.max(population[i, :]) >= pu:
                  chance2classify[np.argmax(population[i, :])] += len(ps) / classesPopDS[1][np.argmax(population[i, :])]
                  cov[np.argmax(population[i, :])] += len(ps)
              chance2cl[str(n)+'_'+str(pu)] = chance2classify
              coverage[str(vLen)+'_'+str(pu)] = cov
            condHperC[str(n)+'_'+str(vLen)] = condHperClass
          MIs[str(n)] = MI_n



            # plt.show()
          # # simulate and test the population of one cluster
          #   population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
          #   # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
          #   clustersNames = range(n)
          #   labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background','elephant',
          #   'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
          #   for clusterIDX in range(n):
          #     idxlist = np.where(clusters.labels_ == clusterIDX)
          #     print(len(idxlist)/len(patches))
          #     GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
          #     classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
          #     for i, pop in enumerate(classesPop[0]):
          #       if b'airplane' in pop and b'255' in pop:
          #         classesPop[0][i] = 0
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'airplane' in pop and b'0' in pop:
          #         classesPop[0][i] = 1
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'dog' in pop and b'255' in pop:
          #         classesPop[0][i] = 2
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'dog' in pop and b'0' in pop:
          #         classesPop[0][i] = 3
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'fire hydrant' in pop and b'255' in pop:
          #         classesPop[0][i] = 6
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'fire hydrant' in pop and b'0' in pop:
          #         classesPop[0][i] = 7
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'elephant' in pop and b'255' in pop:
          #         classesPop[0][i] = 4
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'elephant' in pop and b'0' in pop:
          #         classesPop[0][i] = 5
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'train' in pop and b'255' in pop:
          #         classesPop[0][i] = 8
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #       if b'train' in pop and b'0' in pop:
          #         classesPop[0][i] = 9
          #         population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          #   fig, ax = plt.subplots()
          #   im = ax.imshow(population)
          #   ax.set_yticks(np.arange(len(clustersNames)))
          #   ax.set_xticks(np.arange(len(labelsNames)))
          #   cb = plt.colorbar(im)
          #   plt.savefig('heatmaps_coco_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
          #   plt.close(fig)
            # population[clusterIDX, classesPop[0]] = classesPop[1]
            # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
            #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])

          # plt.show()
          # plt.savefig('keepDaway_newpatches.png')
          # plt.close(fig2)
          # fig, ax = plt.subplots()
          # ax.imshow(population)
          # ax.set_yticks(np.arange(len(clustersNames)), labels=clustersNames)
          # ax.set_xticks(np.arange(len(labelsNames)), labels=labelsNames)
        ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
        random.shuffle(ims)
        fig, axs = plt.subplots(6, 6)
        for i in range(6):
          for j in range(6):
            axs[i, j].imshow(ims[i * 6 + j])
        plt.show()
          # patches = sorted(patches, key=lambda p: p['idx'])
          # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
          # clustering_kmeans = SpectralClustering(n_clusters=50,
          #                                        assign_labels='kmeans',
          #                                        random_state=0,
          #                                        affinity='nearest_neighbors').fit(
          #   np.array([p['patch'].flatten() for p in patches]))
          # clustering_disc = SpectralClustering(n_clusters=50,
          #                                      assign_labels='discretize',
          #                                      random_state=0,
          #                                      affinity='nearest_neighbors').fit(
          #   np.array([p['patch'].flatten() for p in patches]))
          #
          # plt.hist(clustering_disc.labels_, bins=50)
          # plt.hist(clustering_kmeans.labels_, bins=50)


      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument-+++++++++++++++++++++
  if tf.io.gfile.exists(FLAGS.eval_dir):
    tf.compat.v1.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.io.gfile.makedirs(FLAGS.eval_dir)
  if FLAGS.evaluation:
    evaluate()
  else:
    get_representations()




if __name__ == '__main__':
  mp.set_start_method('spawn', force=True)
  tf.compat.v1.app.run()
