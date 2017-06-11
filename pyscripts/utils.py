from collections import defaultdict
import HTMLParser
import copy
import os
import random
import re
import unicodedata

import simplejson as json
import numpy as np

def dspath(ext, ROOT, **kwargs):
    return os.path.join(ROOT,ext)

class Layer(object):
    L1 = 'layer1'
    L2 = 'layer2'
    L3 = 'layer3'
    INGRS = 'det_ingrs'
    GOODIES = 'goodies'

    @staticmethod
    def load(name, ROOT, **kwargs):
        with open(dspath(name + '.json',ROOT, **kwargs)) as f_layer:
            return json.load(f_layer)

    @staticmethod
    def merge(layers, ROOT,copy_base=False, **kwargs):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT, **kwargs) for l in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base

REPLACEMENTS = {
    u'\x91':"'", u'\x92':"'", u'\x93':'"', u'\x94':'"', u'\xa9':'',
    u'\xba': ' degrees ', u'\xbc':' 1/4', u'\xbd':' 1/2', u'\xbe':' 3/4',
    u'\xd7':'x', u'\xae': '',
    '\\u00bd':' 1/2', '\\u00bc':' 1/4', '\\u00be':' 3/4',
    u'\\u2153':' 1/3', '\\u00bd':' 1/2', '\\u00bc':' 1/4', '\\u00be':' 3/4',
    '\\u2154':' 2/3', '\\u215b':' 1/8', '\\u215c':' 3/8', '\\u215d':' 5/8',
    '\\u215e':' 7/8', '\\u2155':' 1/5', '\\u2156':' 2/5', '\\u2157':' 3/5',
    '\\u2158':' 4/5', '\\u2159':' 1/6', '\\u215a':' 5/6', '\\u2014':'-',
    '\\u0131':'1', '\\u2122':'', '\\u2019':"'", '\\u2013':'-', '\\u2044':'/',
    '\\u201c':'\\"', '\\u2018':"'", '\\u201d':'\\"', '\\u2033': '\\"',
    '\\u2026': '...', '\\u2022': '', '\\u2028': ' ', '\\u02da': ' degrees ',
    '\\uf04a': '', u'\xb0': ' degrees ', '\\u0301': '', '\\u2070': ' degrees ',
    '\\u0302': '', '\\uf0b0': ''
}

parser = HTMLParser.HTMLParser()
def prepro_txt(text):
    import urllib

    text = parser.unescape(text)

    for unichar, replacement in REPLACEMENTS.iteritems():
        text = text.replace(unichar, replacement)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')

    try:
        text = urllib.unquote(text).decode('ascii')
    except UnicodeDecodeError:
        pass # if there's an errant %, unquoting will yield an invalid char

    # some extra tokenization
    text = ' - '.join(text.split('-'))
    text = ' & '.join(text.split('&'))

    text = re.sub(r'\\[nt]', ' ', text) # remove over-escaped line breaks and tabs
    text = re.sub(r'\b([^\d\s]+)/(.*)\b', r'\1 / \2', text) # split non-fractions
    text = re.sub(r'\b(.*)/([^\d\s]+)\b', r'\1 / \2', text) # e.g. 350 deg/gas mark
    text = re.sub(r'\s+', ' ', text) # remove extra whitespace

    return text.strip()

def align(haystack, needle):
    haystack = [tok.lower() for tok in haystack]
    needle = [tok.lower() for tok in needle]
    hsz = len(haystack)
    nsz = len(needle)
    s = defaultdict(lambda: (0, 'v'))
    for i in xrange(hsz-1, -1, -1):
        for j in xrange(min(nsz, i+1)-1, max(0, i-hsz+nsz)-1, -1):
            opts = [(s[(i+1, j)][0]-1 if i < hsz-nsz+j else float('-inf'), 'v')]
            if haystack[i] == needle[j]:
                opts.append((s[(i+1, j+1)][0]+1, 'd'))
            s[(i, j)] = max(opts)

    alignment = np.zeros(hsz, dtype='uint8')
    parent = (0, 0)
    for i in xrange(hsz):
        p = s[parent][1]
        alignment[i] = p == 'd'
        parent = (parent[0]+1, parent[1]+(p == 'd'))
    return alignment

def vstack(part_lists, dtype=None):
    part_vecs = {}
    for part, veclist in part_lists.iteritems():
        part_vecs[part] = np.array(veclist, dtype=dtype)
    return part_vecs

def mstack(part_lists, maxlen, dtype=None):
    part_mats = {}
    for part, lists in part_lists.iteritems():
        dim = maxlen[part] if isinstance(maxlen, dict) else maxlen
        part_mat = np.zeros((len(lists), dim), dtype=dtype)
        for i, lst in enumerate(lists):
            part_mat[i, :len(lst)] = lst
        part_mats[part] = part_mat
    return part_mats

def print_counts(counts):
    for item,count in sorted(counts.items(), key=lambda x: -x[1]):
        print('%d\t%s' % (count, item))

def get_partition(trainfrac, valfrac):
    partrand = random.random()
    if partrand < trainfrac:
        return 'train'
    if partrand < trainfrac + valfrac:
        return 'val'
    return 'test'

def load_vocab(p, offset=1):
    with open(p) as f_vocab:
        vocab = {w.rstrip(): i+offset for i, w in enumerate(f_vocab)}
    return vocab
