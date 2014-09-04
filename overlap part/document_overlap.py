#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.cluster import DBSCAN
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


# Define News class

class New:
    def __init__(self, number, title, time, category, content, url):
        self.number = number
        self.title = title
        self.time = time
        self.category = category
        self.content = content
        self.url = url
        self.similarity = []
        

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

news = []

for root, dirs, files in os.walk(path):
    # print root
    for j,f in enumerate(files):
        news.append([])
        print("================================   " +str(j)+ "   ================================")
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
        news[j] = New(j, title, date, -1, doc, url)


        fileread = lambda filename: open(filename, "rb").read()
        coding = chardet.detect(fileread(file_path))['encoding']
		# {'confidence': 0.99, 'encoding': 'utf-8'}
        news[j].content = news[j].content.decode( coding )

        
        seg_list = jieba.cut_for_search(news[j].content)  # 搜索引擎模式
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

    # http://stackoverflow.com/questions/20563239/truncatedsvd-explained-variance
    svd = TruncatedSVD().fit(X)
    X_proj = svd.transform(X)
    explained_variances = np.var(X_proj, axis=0) / np.var(X, axis=0).sum()

    # explained_variance = svd.explained_variance_ratio_.sum() # broken
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variances[0] * 100)))

    print()


#############################################################################
# Do the actual clustering

if opts.minibatch:
    print("MiniBatchKMeans")
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=True)
else:
    # print("KMeans")
    # km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
    #             verbose=True)
    print("DBSCAN")
    km = DBSCAN(eps=0.3, min_samples=10)

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

for i in range(len(news)):
    news[i].category = labels[i]

from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(news)):
    news[i].similarity = cosine_similarity(X[i:i+1], X)[0]
    cs = news[i].similarity

for i in range(len(news)):
    print(news[i].number, news[i].title, news[i].time, news[i].category, news[i].url, news[i].similarity)

from heapq import nlargest

# ======================== overlap method 1 =========================================
print ("overlap method 1")

New_similarity = []
cross = 0
crosspoint = []
k = len(news) # the similarity of first len(news) news would be the sam

for i in range(len(news)):
    New_similarity.extend(news[i].similarity)

New_similarity = np.array(New_similarity)

while cross < 5: # control how many crosspoints 
    k = k+2 # it is doubled side
    topk_similarity = nlargest(k, enumerate(New_similarity), key=lambda x: x[1])[-1]
    
#     print (topk_similarity)
#     print (topk_similarity[0]/len(news)) # the article I belong
#     print (topk_similarity[0]%len(news)) # the article I similar with
#     print (labels[topk_similarity[0]/len(news)])
#     print (labels[topk_similarity[0]%len(news)])
    
    if labels[topk_similarity[0]/len(news)] != labels[topk_similarity[0]%len(news)]:
        cross += 1
        crosspoint.append(topk_similarity)
        
        print (topk_similarity)
        print (news[(topk_similarity[0]/len(news))].content) # the article I belong
        print (news[(topk_similarity[0]%len(news))].content) # the article I similar with
        print (topk_similarity[0]/len(news))
        print (topk_similarity[0]%len(news))

print (crosspoint)

for point in crosspoint:
    print ((point[0]/len(news),point[0]%len(news)))


# ======================== overlap method 2 =========================================
print ("overlap method 2")

topk_similarity_m2 = nlargest(100+len(news), enumerate(New_similarity), key=lambda x: x[1])
crosspoint_m2 = []

for sim in topk_similarity_m2[len(news):]:
    crosspoint_m2.append((sim[0]/len(news),sim[0]%len(news)))


import sys, os
from copy import copy
from operator import itemgetter
from heapq import heappush, heappop
from collections import defaultdict
from itertools import combinations # requires python 2.6+
from optparse import OptionParser

def swap(a,b):
    if a > b:
        return b,a
    return a,b


def Dc(m,n):
    """partition density"""
    try:
        return m*(m-n+1.0)/(n-2.0)/(n-1.0)
    except ZeroDivisionError: # numerator is "strongly zero"
        return 0.0


class HLC:
    def __init__(self,adj,edges):
        self.adj   = adj
        self.edges = edges
        self.Mfactor  = 2.0 / len(edges)
        self.edge2cid = {}
        self.cid2nodes,self.cid2edges = {},{}
        self.initialize_edges() # every edge in its own comm
        self.D = 0.0 # partition density
    
    def initialize_edges(self):
        for cid,edge in enumerate(self.edges):
            edge = swap(*edge) # just in case
            self.edge2cid[edge] = cid
            self.cid2edges[cid] = set([edge])
            self.cid2nodes[cid] = set( edge )
    
    def merge_comms(self,edge1,edge2):
        cid1,cid2 = self.edge2cid[edge1],self.edge2cid[edge2]
        if cid1 == cid2: # already merged!
            return
        m1,m2 = len(self.cid2edges[cid1]),len(self.cid2edges[cid2])
        n1,n2 = len(self.cid2nodes[cid1]),len(self.cid2nodes[cid2])
        Dc1, Dc2 = Dc(m1,n1), Dc(m2,n2)
        if m2 > m1: # merge smaller into larger
            cid1,cid2 = cid2,cid1
        
        self.cid2edges[cid1] |= self.cid2edges[cid2]
        for e in self.cid2edges[cid2]: # move edges,nodes from cid2 to cid1
            self.cid2nodes[cid1] |= set( e )
            self.edge2cid[e] = cid1
        del self.cid2edges[cid2], self.cid2nodes[cid2]
        
        m,n = len(self.cid2edges[cid1]),len(self.cid2nodes[cid1]) 
        Dc12 = Dc(m,n)
        self.D = self.D + ( Dc12 -Dc1 - Dc2) * self.Mfactor # update partition density
    
    def single_linkage(self,threshold=None):
        """docstring goes here..."""
        print("clustering...")
        self.list_D = [(1.0,0.0)] # list of (S_i,D_i) tuples...
        self.best_D = 0.0
        self.best_S = 1.0 # similarity threshold at best_D
        self.best_P = None # best partition, dict: edge -> cid
        
        H = similarities( self.adj ) # min-heap ordered by 1-s
        S_prev = -1
        for oms,eij_eik in H:
            S = 1-oms # remember, H is a min-heap
            if S < threshold:
                break
                
            if S != S_prev: # update list
                if self.D >= self.best_D: # check PREVIOUS merger, because that's
                    self.best_D = self.D  # the end of the tie
                    self.best_S = S
                    self.best_P = copy(self.edge2cid) # slow...
                self.list_D.append( (S,self.D) )
                S_prev = S
            self.merge_comms( *eij_eik )
        
        self.list_D.append( (0.0,self.list_D[-1][1]) ) # add final val
        if threshold != None:
            return self.edge2cid, self.D
        return self.best_P, self.best_S, self.best_D, self.list_D
    


def similarities(adj):
    """Get all the edge similarities. Input dict maps nodes to sets of neighbors.
    Output is a list of decorated edge-pairs, (1-sim,eij,eik), ordered by similarity.
    """
    print("computing similarities...")
    i_adj = dict( (n,adj[n] | set([n])) for n in adj)  # node -> inclusive neighbors
    min_heap = [] # elements are (1-sim,eij,eik)
    for n in adj: # n is the shared node
        if len(adj[n]) > 1:
            for i,j in combinations(adj[n],2): # all unordered pairs of neighbors
                edge_pair = swap( swap(i,n),swap(j,n) )
                inc_ns_i,inc_ns_j = i_adj[i],i_adj[j] # inclusive neighbors
                S = 1.0 * len(inc_ns_i&inc_ns_j) / len(inc_ns_i|inc_ns_j) # Jacc similarity...
                heappush( min_heap, (1-S,edge_pair) )
    return [ heappop(min_heap) for i in xrange(len(min_heap)) ] # return ordered edge pairs


def read_edgelist(nodetype=str):
    adj = defaultdict(set)
    edges = set()
    for line in crosspoint_m2:
        L = line
        ni,nj = nodetype(L[0]),nodetype(L[1]) # other columns ignored
        if ni != nj: # skip any self-loops...
            edges.add( swap(ni,nj) )
            adj[ni].add(nj)
            adj[nj].add(ni) # since undirected
    return dict(adj), edges


def write_edge2cid(e2c,delimiter=","):
    # write edge2cid three-column file
    c2c = dict( (c,i+1) for i,c in enumerate(sorted(list(set(e2c.values())))) ) # ugly...
    # for e,c in sorted(e2c.iteritems(), key=itemgetter(1)):
        # print ( "%s%s%s%s%s\n" % (str(e[0]),delimiter,str(e[1]),delimiter,str(c2c[c])) )
    
    cid2edges,cid2nodes = defaultdict(set),defaultdict(set) # faster to recreate here than
    for edge,cid in e2c.iteritems():                        # to keep copying all dicts
        cid2edges[cid].add( edge )                          # during the linkage...
        cid2nodes[cid] |= set(edge)
    cid2edges,cid2nodes = dict(cid2edges),dict(cid2nodes)
    
    # write list of edges for each comm, each comm on its own line
    for cid in sorted(cid2edges.keys()):
        nodes,edges = map(str,cid2nodes[cid]), ["%s,%s" % (ni,nj) for ni,nj in cid2edges[cid]]
        if (len(edges) == 1):
            print ( " ".join(edges) )
        # print ( " ".join([str(cid)] + nodes) );


if __name__ == '__main__':

    print("# loading input...")
    adj,edges = read_edgelist()
    
    edge2cid,S_max,D_max,list_D = HLC( adj,edges ).single_linkage()
    # for s,D in list_D:
    #     print(s, D)
    # print("# D_max = %f\n# S_max = %f" % (D_max,S_max))
    write_edge2cid( edge2cid )

        
