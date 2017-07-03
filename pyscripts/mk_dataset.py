#!/usr/bin/env python
import random
import pickle
import h5py
import numpy as np
from proc import *
from tqdm import *
import torchfile
import time
import utils
import os
from params import get_parser

def center_crop(imsize,im):

    width, height,c = im.shape   # Get dimensions

    left = (width - imsize)/2
    top = (height - imsize)/2
    right = (width + imsize)/2
    bottom = (height + imsize)/2

    return im[left:right,top:bottom,:]

def check_partitions(params,dataset,ingr_vocab,remove_ids):
    '''
    Gets size of partitions in dataset
    '''
    n_samples = {'train':0,'val':0,'test':0}

    for i,entry in enumerate(dataset):
        ingr_detections = detect_ingrs(entry, ingr_vocab)
        ningrs = len(ingr_detections)
        ninstrs = len(entry['instructions'])

        imgs = entry.get('images')
        #if ninstrs >= params.maxlen or ningrs >= params.maxlen or ningrs == 0 or not imgs: # if dataset is clean use this
        if ninstrs >= params.maxlen or ningrs >= params.maxlen or ningrs == 0 or not imgs or remove_ids.get(entry['id']):
            continue
        partition = entry['partition']
        n_samples[partition]+=len(imgs)

    return n_samples

def get_st(file):
    info = torchfile.load(file)

    ids = info['ids']

    imids = []
    for i,id in enumerate(ids):
        imids.append(''.join(chr(i) for i in id))

    st_vecs = {}
    st_vecs['encs'] = info['encs']
    st_vecs['rlens'] = info['rlens']
    st_vecs['rbps'] = info['rbps']
    st_vecs['ids'] = imids

    print(np.shape(st_vecs['encs']),len(st_vecs['rlens']),len(st_vecs['rbps']),len(st_vecs['ids']))
    return st_vecs

parser = get_parser()
params = parser.parse_args()

IMPATH = os.path.join(params.dataset,'images')
DATASET = params.dataset

# don't use this file once dataset is clean
with open('./remove1M.txt','r') as f:
    remove_ids = {w.rstrip(): i for i, w in enumerate(f)}

t = time.time()
print ("Loading skip-thought vectors...")

if params.suffix == '1M':
    sthdir = params.stvecs
    st_vecs_train = get_st(os.path.join(sthdir, 'encs_train_1024.t7'))
    st_vecs_val = get_st(os.path.join(sthdir, 'encs_val_1024.t7'))
    st_vecs_test = get_st(os.path.join(sthdir, 'encs_test_1024.t7'))

    st_vecs = {'train':st_vecs_train,'val':st_vecs_val,'test':st_vecs_test}
    stid2idx = {'train':{},'val':{},'test':{}}

    for part in ['train','val','test']:
        for i,id in enumerate(st_vecs[part]['ids']):
            stid2idx[part][id] = i
else:
    stvecs = np.load('../data/text/encoded_instructions.npy')

print ("Done.",time.time() - t)

print('Loading dataset.')
dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS],DATASET)

print('Loading ingr vocab.')
with open(params.vocab) as f_vocab:
    ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f_vocab)} # +1 for lua
    ingr_vocab['</i>'] = 1

with open('../data/classes'+params.suffix+'.pkl','rb') as f:
    class_dict = pickle.load(f)
    id2class = pickle.load(f)

print('Image path is:',IMPATH)
f_ds = h5py.File(params.h5_data, 'w')
imsize = params.imsize
n_samples = check_partitions(params,dataset,ingr_vocab,remove_ids)
print('H5 file is',params.h5_data)

print (n_samples)
train_ims = f_ds.create_dataset('ims_train',(n_samples['train'],3,imsize,imsize),dtype=np.uint8)
val_ims = f_ds.create_dataset('ims_val',(n_samples['val'],3,imsize,imsize),dtype=np.uint8)
test_ims = f_ds.create_dataset('ims_test',(n_samples['test'],3,imsize,imsize),dtype=np.uint8)

st_ptr = 0
numfailed = 0
idx = {'train':0,'val':0,'test':0}
ids = { 'train': [], 'val': [], 'test': [] }
image_names = { 'train': [], 'val': [], 'test': [] }
dsvecs_lists = { 'train': [], 'val': [], 'test': [] }
dsingrs_lists = { 'train': [], 'val': [], 'test': [] }
recipe_lens = { 'train': [], 'val': [], 'test': [] }
ingr_lens = { 'train': [], 'val': [], 'test': [] }
rbp = { 'train': [], 'val': [], 'test': [] } # base pointer into rvecs
classes = {'train' : [], 'val': [], 'test': []}
num_ims = {'train' : [], 'val': [], 'test': []}
im_pos = {'train' : [], 'val': [], 'test': []}

print('Assembling dataset.')
num_samples = {'train':0,'val':0,'test':0}
for i,entry in tqdm(enumerate(dataset)):

    ninstrs = len(entry['instructions'])

    ingr_detections = detect_ingrs(entry, ingr_vocab)
    ningrs = len(ingr_detections)
    imgs = entry.get('images')

    # Samples to remove
    #if ninstrs >= params.maxlen or ningrs >= params.maxlen or ningrs == 0 or not imgs: # if dataset is clean use this
    if ninstrs >= params.maxlen or ningrs >= params.maxlen or ningrs == 0 or not imgs or remove_ids.get(entry['id']):
        st_ptr += ninstrs
        continue

    partition = entry['partition']
    ids[partition].append(entry['id'])
    classes[partition].append(class_dict[entry['id']]+1) # start at 1 for lua

    # start position for skip-thoughts
    part_rbp = rbp[partition]
    if len(part_rbp) == 0:
        part_rbp.append(1) # 1 for lua
    else:
        rbp[partition].append(rbp[partition][-1] + recipe_lens[partition][-1]) #starting position is the previous one plus the previous length

    if params.suffix == '1M':
        stpos = stid2idx[partition][entry['id']] #select the sample corresponding to the index in the skip-thoughts data
        beg = st_vecs[partition]['rbps'][stpos] - 1 # minus 1 because it was saved in lua
        end = beg + st_vecs[partition]['rlens'][stpos]
        dsvecs_lists[partition].append(st_vecs[partition]['encs'][beg:end]) # select the collection of skip-thought vectors for that sample
        recipe_lens[partition].append(st_vecs[partition]['rlens'][stpos])
    else:
        dsvecs_lists[partition].append(stvecs[st_ptr:st_ptr+ninstrs])
        recipe_lens[partition].append(ninstrs)
        st_ptr += ninstrs

    dsingrs_lists[partition].append(ingr_detections)
    ingr_lens[partition].append(ningrs)

    num_ims[partition].append(min(params.maxims,len(imgs)))
    impos_list = []
    for imid in range(min(params.maxims,len(imgs))):
        
        image_id = entry['images'][imid]['id']
        # images were arranged in a four-level hierarchy corresponding to the
        # first four digits of the image id. For example: `val/e/f/3/d/ef3dc0de11.jpg`
        imname = os.path.join(IMPATH,partition,image_id[0],image_id[1],
                                               image_id[2],image_id[3],
                                               image_id)
        image_names[partition].append(imname)
        img,fail = process_image(imname,imsize)
        img = center_crop(imsize,img)
        numfailed+=fail

        if partition=='train':
            train_ims[idx[partition],:,:,:] = img.transpose(2, 0, 1)
        elif partition=='val':
            val_ims[idx[partition],:,:,:] = img.transpose(2, 0, 1)
        elif partition=='test':
            test_ims[idx[partition],:,:,:] = img.transpose(2, 0, 1)
        impos_list.append(idx[partition]+1) # +1 for lua
        idx[partition]+=1
    im_pos[partition].append(impos_list)

dsvecs = {}
dsingrs = {}
dsimpos = {}
for part in ['train', 'val', 'test']:
    # instructions
    dsvecs[part] = np.concatenate(dsvecs_lists[part])

    # ingredients
    dsingrs_vec = np.zeros((len(dsingrs_lists[part]), params.maxlen), dtype='uint16')
    for i, ingrlist in enumerate(dsingrs_lists[part]):
        dsingrs_vec[i, :len(ingrlist)] = ingrlist
    dsingrs[part] = dsingrs_vec

    # images
    im_pos_vec = np.zeros((len(im_pos[part]), params.maxims), dtype='uint32')
    for i, imlist in enumerate(im_pos[part]):
        im_pos_vec[i, :len(imlist)] = imlist
    dsimpos[part] = im_pos_vec


print('Writing out data.')
for part in ids:
    f_ds.create_dataset('/ids_%s' % part, data=ids[part])
    f_ds.create_dataset('/classes_%s' % part, data=classes[part])
    f_ds.create_dataset('/imnames_%s' % part, data=image_names[part])
    f_ds.create_dataset('/stvecs_%s' % part, data=dsvecs[part])
    f_ds.create_dataset('/ingrs_%s' % part, data=dsingrs[part])
    f_ds.create_dataset('/rlens_%s' % part, data=recipe_lens[part])
    f_ds.create_dataset('/ilens_%s' % part, data=ingr_lens[part])
    f_ds.create_dataset('/rbps_%s' % part, data=rbp[part])
    f_ds.create_dataset('/numims_%s' % part, data=num_ims[part])
    f_ds.create_dataset('/impos_%s' % part, data=dsimpos[part])

f_ds.close()
