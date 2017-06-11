import numpy as np
import utils
from PIL import Image
from tqdm import *
import os
from proc import *
import random
from params import get_parser

'''
Selects subset of images for segmentation
'''

create = False
parser = get_parser()
params = parser.parse_args()
dataset = params.dataset

if create:
    '''
    Filter out small images and duplicates and saves a list in
    files/kept_for_segmentation.txt with ids to keep
    '''
    kept = open('files/kept_for_segmentation.txt','w') # outfile
    impath = os.path.join(params.dataset,'images')
    remove_ids = []
    with open('files/remove1M.txt','r') as f: # text & image duplicates (ids to remove)
        ids = f.readlines()
        for id in ids:
            remove_ids.append(id.rstrip())

    minsize = 500 #images smaller than 500x500 are discarded
    removed_imsize = 0
    print remove_ids[0:10]
    dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS],dataset)

    kept_ids = []
    for i,entry in tqdm(enumerate(dataset)):

        imgs = entry.get('images')

        if not imgs or entry['id'] in remove_ids:
            continue

        imname = entry['images'][0]['id']
        im = Image.open(os.path.join(impath,imname))
        w,h = im.size # (width,height) tuple
        #w,h = get_image_size(os.path.join(impath,imname))
        if w < minsize or h < minsize:
            removed_imsize+=1
            continue
        kept_ids.append(entry['id'])

    print removed_imsize
    print len(kept_ids)
    kept.write("\n".join(kept_ids))
    kept.close()

else:
    print "Selecting images based on ingredients..."
    vocab = 'files/vocab.txt'
    kept_ids = []
    with open('files/kept_for_segmentation.txt','r') as f: # text & image duplicates (ids to remove)
        ids = f.readlines()
        for id in ids:
            kept_ids.append(id.rstrip())

    dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS],dataset)

    ind2idx = {}
    for i,entry in enumerate(dataset):
        ind2idx[entry['id']] = i

    print('Loading ingr vocab.')
    with open(vocab) as f_vocab:
        ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f_vocab)} # +1 for lua
        ingr_vocab['</i>'] = 1
    ingr_vocab_inv = {v: k for k, v in ingr_vocab.iteritems()}

    allingrs = []
    for id in kept_ids:
        entry = dataset[ind2idx[id]]
        ingr_detections = detect_ingrs(entry, ingr_vocab)
        for ingr in ingr_detections:
            allingrs.append(ingr_vocab_inv[ingr])


    from collections import Counter
    N = 1001
    prob = 0.7
    top_k = Counter(allingrs).most_common(N)
    names,counts = zip(*top_k[1:])

    with open('files/seg/ingrs1k.txt','w') as f:
        for name,count in zip(names,counts):
            f.write(name + '\t' + str(count) + '\n')

    max_num = counts[-1]
    print "Max number per class:", max_num
    ingr_counts = {}
    for name in names:
        ingr_counts[name]=0

    topkinds = []
    for name in names[0:10]:
        topkinds.append(ingr_vocab[name])
    print topkinds

    ids = []
    imids = []
    for id in tqdm(kept_ids):
        entry = dataset[ind2idx[id]]
        ingr_detections = detect_ingrs(entry, ingr_vocab)
        img = entry['images'][0]['id']

        # if one of the most freq ingredients exists, lower prob of adding sample
        hastopk = len(set(ingr_detections).intersection(topkinds))
        if hastopk>0:
            p = random.random()
        else:
            p = 1
        if p > prob:
            for i,name in enumerate(names[::-1]):
                if not ingr_counts[name]>=max_num and ingr_vocab[name] in ingr_detections and random.random()>i/(N-1):
                    imids.append(img)
                    ids.append(entry['id'])
                    for ingr in ingr_detections:
                        if ingr_vocab_inv[ingr] in ingr_counts:
                            ingr_counts[ingr_vocab_inv[ingr]]+=1
                    break

    import operator
    sorted_ingrs = sorted(ingr_counts.items(), key=operator.itemgetter(1),reverse=True)
    with open('files/seg/ingrs1k_filtered.txt','w') as f:
        for name,count in sorted_ingrs:
            f.write(name + '\t' + str(count) + '\n')
    with open('files/seg/ids_seg.txt','w') as f:
        f.write("\n".join(ids))
    with open('files/seg/ims_seg.txt','w') as f:
        f.write("\n".join(imids))

    print len(imids)
