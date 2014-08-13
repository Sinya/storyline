#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

# """
# Display Information about a Google Calendar(Testing)

#  -u --user login     Google Login
#  -p --pass password  Google Password
#  -d --debug          Set DEBUG = True
#  -h --help           Display this help
# """

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

# jieba
import os
import jieba
import chardet
import jieba.analyse
import jieba.posseg as pseg

# load text files
import os
import chardet


# The apart text of corpus

stop_words = []
file = open("stopword" ,"rb")
for line in file:
    stop_words.append(line[:len(line)-1])
    
# print stop_words

def tokenizer(stop_words,s):
    from nltk.tokenize import WhitespaceTokenizer
    tok = s.split(" ")
    # print stop_words
    # remove stop words
    result = ""
    for w in tok:
        if w.strip() not in stop_words and w.strip() != "":
            result += w.strip()
            result += " "
            
    return result


jieba.load_userdict("dict.txt")
path = "fat_news/"
corpus = []

for root, dirs, files in os.walk(path):
    # print root
    for j,f in enumerate(files):
        print("====================================   " +str(j)+ "   ====================================")
        doc = ""
        date = 0
        url = ""
        title = ""

        file_path = os.path.join(root, f)
        with open( file_path ) as op:
            for i,line in enumerate(op):
                if i is 0:
                    url = line
                elif i is 2:
                    date = line # use later
                elif i is 4:
                    title = line
                else:
                    doc += line
        fileread = lambda filename: open(filename, "rb").read()
        coding = chardet.detect(fileread(file_path))['encoding']
		# {'confidence': 0.99, 'encoding': 'utf-8'}
        doc = doc.decode( coding )

        
        seg_list = jieba.cut_for_search(doc)  # 搜索引擎模式
#         print(" ".join(seg_list).encode('utf-8'))
        result_list = " ".join(seg_list).encode('utf-8')
        result_list = tokenizer(stop_words,result_list)
        corpus.append(result_list)
        

# print(corpus)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")


# __doc__ is a global variable that contains the documentation string of your script
print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
# Load some categories from the training set
# categories = [
#     'alt.atheism',
#     'talk.religion.misc',
#     'comp.graphics',
#     'sci.space',
# ]
# Uncomment the following to do the analysis on all the categories
categories = None

# print()
# print("Loading 20 newsgroups dataset for categories:")
# print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

# print("%d documents" % len(dataset.data))
# print("%d categories" % len(dataset.target_names))
# print()

# labels = dataset.target
# true_k = np.unique(labels).shape[0]
from numpy import array

n_digits = 3
n_samples, n_features = (25,1927)
labels = array([0,1,2,1,1,2,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,2,1])
true_k = np.unique(labels).shape[0]

# print(true_k)
# print(data)

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(corpus)

# print(X)


print("done in %fs" % (time() - t0))
# n_samples: how many articles are there
# n_features: how many different words in all articles are there
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    lsa = make_pipeline(svd, Normalizer(copy=False))

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


###############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)
#    km = KMeans(n_clusters=true_k, init='k-means++',  n_init=1,
#                verbose=opts.verbose)


print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))

print("Silhouette Coefficient: %0.3f"
     % metrics.silhouette_score(X, labels, sample_size=None))

print(labels)

print()

if not (opts.n_components or opts.use_hashing):
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
