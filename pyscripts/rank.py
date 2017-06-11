import os
import random
from sklearn.preprocessing import normalize
import numpy as np
import utils
import torchfile
from params import get_parser

parser = get_parser()
params = parser.parse_args()

random.seed(params.seed)
DATA_ROOT = params.test_feats
partition = params.partition

img_embs = partition + '_img_embs.t7'
instr_embs = partition + '_instr_embs.t7'
test_ids = partition + '_ids.t7'

im_vecs = np.array(torchfile.load(os.path.join(DATA_ROOT,img_embs)))
instr_vecs = np.array(torchfile.load(os.path.join(DATA_ROOT,instr_embs)))
names = np.array(torchfile.load(os.path.join(DATA_ROOT,test_ids)))

# Sort based on names to always pick same samples for medr
names_str = []
for i in range(names.shape[0]):
    names_str.append(''.join(chr(k) for k in names[i]).split('\x00')[0])
names = np.array(names_str)
idxs = np.argsort(names)
names = names[idxs]
im_vecs = normalize(im_vecs)[idxs]
instr_vecs = normalize(instr_vecs)[idxs]

# Ranker
N = params.medr
idxs = range(N)

glob_rank = []
glob_recall = {1:0.0,5:0.0,10:0.0}
for i in range(10):

    ids = random.sample(xrange(0,len(names)), N)
    im_sub = im_vecs[ids,:]
    instr_sub = instr_vecs[ids,:]
    ids_sub = names[ids]

    if params.embedding == 'image':
        sims = np.dot(im_sub,instr_sub.T) # for im2recipe
    else:
        sims = np.dot(instr_sub,im_sub.T) # for recipe2im

    med_rank = []
    recall = {1:0.0,5:0.0,10:0.0}

    for ii in idxs:

        name = ids_sub[ii]
        # get a column of similarities
        sim = sims[ii,:]

        # sort indices in descending order
        sorting = np.argsort(sim)[::-1].tolist()

        # find where the index of the pair sample ended up in the sorting
        pos = sorting.index(ii)

        if (pos+1) == 1:
            recall[1]+=1
        if (pos+1) <=5:
            recall[5]+=1
        if (pos+1)<=10:
            recall[10]+=1

        # store the position
        med_rank.append(pos+1)

    for i in recall.keys():
        recall[i]=recall[i]/N

    med = np.median(med_rank)
    print "median", med

    for i in recall.keys():
        glob_recall[i]+=recall[i]
    glob_rank.append(med)

for i in glob_recall.keys():
    glob_recall[i] = glob_recall[i]/10
print "Mean median", np.average(glob_rank)
print "Recall", glob_recall
