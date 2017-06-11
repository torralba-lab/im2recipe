import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import json
from params import get_parser
from sklearn.preprocessing import normalize
import random
import utils
import torchfile
from scipy.misc import imread,imresize
import sys
from shutil import copyfile


IMPATH = '../../data/recipe800k/images/'

def read_image(impath):
    im = Image.open(impath)
    im = im.resize((224,224))
    return im

def load_layer(json_file):
    with open(json_file) as f_layer:
        return json.load(f_layer)


ch = {'sushi':[],'pie':[],'pizza':[],'lasagna':[],'soup':[],'burger':[],
      'pasta':[],'salad':[],'smoothie':[],'cookie':[]}

parser = get_parser()
params = parser.parse_args()

#random.seed(params.seed)

DATA_ROOT = params.test_feats
IMPATH = os.path.join(params.dataset,'images')
partition = params.partition

img_embs = partition + '_img_embs.t7'
instr_embs = partition + '_instr_embs.t7'
test_ids = partition + '_ids.t7'

im_vecs = np.array(torchfile.load(os.path.join(DATA_ROOT,img_embs)))
instr_vecs = np.array(torchfile.load(os.path.join(DATA_ROOT,instr_embs)))
names = np.array(torchfile.load(os.path.join(DATA_ROOT,test_ids)))

im_vecs = normalize(im_vecs)
instr_vecs = normalize(instr_vecs)

# load dataset
print('Loading dataset.')
dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS],params.dataset)
print "Done."
idx2ind = {} #sample id to position in dataset
for i in range(len(dataset)):
    idx2ind[dataset[i]['id']] = i

names_str = []
for name in names:
    names_str.append(''.join(chr(k) for k in name).split('\x00')[0])
names = names_str

idx2ind_test = {} # sample id to position in embedding matrix
for i in range(len(names)):
    idx2ind_test[names[i]] = i

for i,name in enumerate(names):
    title = dataset[idx2ind[name]]['title']
    for j,v in ch.iteritems():
        if j in title.lower():
            ch[j].append(i)

q_vecs = im_vecs
d_vecs = instr_vecs

# Ranker
N = 1000
idxs = range(N)
K = 8 # number of subplots
MAXLEN_INGRS = 20
SPN = 6 # text separation (y)
fsize = 20 #text size
max_n = 8 #max number of instructions & ingredients in the list
ref_y = 225

ids_sub = names
sims = np.dot(q_vecs,d_vecs.T) # for im2recipe

sample_ids = [ch['pizza'][8],ch['sushi'][5],ch['lasagna'][0],
              ch['soup'][4],ch['smoothie'][3],ch['smoothie'][8],
              ch['cookie'][0],ch['salad'][3],ch['sushi'][0],ch['burger'][3],
              ch['pizza'][5],ch['sushi'][2],ch['sushi'][12],ch['soup'][2],
              ch['cookie'][1],ch['pizza'][10],ch['burger'][7],ch['salad'][11],
              ch['cookie'][2],ch['pizza'][0]]
num_plot = len(sample_ids)

plt.figure(0,figsize = (40,40))
savedir = 'examples/'

for pos,i in enumerate(sample_ids):

    # get a column of similarities
    sim_i2r = sims[i,:]
    sim_r2i = sims[:,i]

    # sort indices in descending order
    ind_pred_i2r = ids_sub[np.argsort(sim_i2r)[::-1].tolist()[0]]
    ind_pred_r2i = ids_sub[np.argsort(sim_r2i)[::-1].tolist()[0]]

    ind_true = ids_sub[i]

    # find sample in database
    pred_i2r = dataset[idx2ind[ind_pred_i2r]]
    pred_r2i = dataset[idx2ind[ind_pred_r2i]]
    true_entry = dataset[idx2ind[ind_true]]

    imname = os.path.join(IMPATH,true_entry['images'][0]['id'])
    copyfile(imname,os.path.join(savedir,str(pos)+'_query_im.jpg'))
    # true image
    img = imresize(imread(imname),(224,224))
    ax = plt.subplot2grid((num_plot,K), (pos,0))
    ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # query ingrs
    ingrs = true_entry['ingredients']
    ax2 = plt.subplot2grid((num_plot,K), (pos,2),colspan=1)
    ax2.axis([0, 250, 0, 250])
    x = 10
    y = ref_y
    with open(os.path.join(savedir,str(pos)+'_query_title.txt'),'w') as f:
        f.write("%s\n" % true_entry['title'])
    with open(os.path.join(savedir,str(pos)+'_query_ingrs.txt'),'w') as f:
        for ingr in ingrs:
            f.write("%s\n" % ingr['text'])
    instrs = true_entry['instructions']
    with open(os.path.join(savedir,str(pos)+'_query_instrs.txt'),'w') as f:
        for instr in instrs:
            f.write("%s\n" % instr['text'])
    for q,ingr in enumerate(ingrs):
        text = ingr['text']
        text = text[0:MAXLEN_INGRS]
        ax2.text(x,y, text,fontsize=fsize, ha='left')
        y-=fsize+SPN
        if q == max_n:
            break
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    # retrieved ingrs
    ingrs = pred_i2r['ingredients']
    instrs = pred_i2r['instructions']
    with open(os.path.join(savedir,str(pos)+'_retr_title.txt'),'w') as f:
        f.write("%s\n" % pred_i2r['title'])
    with open(os.path.join(savedir,str(pos)+'_retr_ingrs.txt'),'w') as f:
        for ingr in ingrs:
            f.write("%s\n" % ingr['text'])
    with open(os.path.join(savedir,str(pos)+'_retr_instrs.txt'),'w') as f:
        for instr in instrs:
            f.write("%s\n" % instr['text'])
    ax3 = plt.subplot2grid((num_plot,K), (pos,1),colspan=1)
    ax3.axis([0, 250, 0, 250])
    x = 10
    y = ref_y
    for q,ingr in enumerate(ingrs):
        text = ingr['text']
        text = text[0:MAXLEN_INGRS]
        ax3.text(x,y, text,fontsize=fsize, ha='left')
        y-=fsize+SPN
        if q == max_n:
            break
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    # retrieved image
    imname = os.path.join(IMPATH,pred_r2i['images'][0]['id'])
    copyfile(imname,os.path.join(savedir,str(pos)+'_retr_im.jpg'))
    # image
    img = imresize(imread(imname),(224,224))
    ax4 = plt.subplot2grid((num_plot,K), (pos,3))
    ax4.imshow(img)
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)


plt.savefig('../data/figs/im2recipe.png')
plt.close()
