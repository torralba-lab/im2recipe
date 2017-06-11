from scipy.misc import imread, imresize
import numpy as np
def detect_ingrs(recipe, vocab):
    try:
        ingr_names = [ingr['text'] for ingr in recipe['ingredients'] if ingr['text']]
    except:
        ingr_names = []
        print "Could not load ingredients! Moving on..."

    detected = set()
    for name in ingr_names:
        name = name.replace(' ','_')
        name_ind = vocab.get(name)
        if name_ind:
            detected.add(name_ind)
        '''
        name_words = name.lower().split(' ')
        for i in xrange(len(name_words)):
            name_ind = vocab.get('_'.join(name_words[i:]))
            if name_ind:
                detected.add(name_ind)
                break
        '''

    return list(detected) + [vocab['</i>']]

def process_image(impath,imsize):
    try:
        img = imread(impath)
        if img.ndim == 2: #grayscale
            img = img[:,:,None][:,:,[0,0,0]]
        H0, W0 = img.shape[0], img.shape[1]

        img = imresize(img, float(imsize) / min(H0, W0))
        fail = 0
    except:
        print "Could not load image...Using black one instead."
        img = np.zeros((imsize,imsize,3))
        fail =1

    return img,fail

def read_image(filename):
    img = imread(filename)
    if img.ndim == 2:
        img = img[:, :, None][:, :, [0, 0, 0]]

    img = imresize(img, (224,224))
    return img
