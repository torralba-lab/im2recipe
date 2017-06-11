import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='tri-joint parameters')

    parser.add_argument('-partition',   dest='partition',       default = 'test')
    parser.add_argument('-nlosscurves', dest='nlosscurves',     default = 3,        type=int)
    parser.add_argument('-embedding',   dest='embedding',       default = 'image')
    parser.add_argument('-medr',        dest='medr',            default = 1000,     type=int)
    parser.add_argument('-tsamples',    dest='tsamples',        default = 20,       type=int)
    parser.add_argument('-maxlen',      dest='maxlen',          default = 20,       type=int)
    parser.add_argument('-maxims',      dest='maxims',          default = 5,        type=int)
    parser.add_argument('-seed',        dest='seed',            default = 42,       type=int)
    parser.add_argument('-imsize',      dest='imsize',          default = 256,      type=int)
    parser.add_argument('-dispfreq',    dest='dispfreq',        default = 1000,     type=int)
    parser.add_argument('-valfreq',     dest='valfreq',         default = 10000,    type=int)
    parser.add_argument('-test_feats',  dest='test_feats',      default = '../results/')

    # new dataset 1M
    parser.add_argument('-f101_cats',   dest='f101_cats',       default = '../data/food101_classes_renamed.txt')
    parser.add_argument('-vocab',       dest='vocab',           default = '../data/text/vocab.txt')
    parser.add_argument('-stvecs',      dest='stvecs',          default = '../data/text/')
    parser.add_argument('-dataset',     dest='dataset',         default = '../data/recipe1M/')
    parser.add_argument('-suffix',      dest='suffix',          default = '1M')
    parser.add_argument('-h5_data',     dest='h5_data',         default = '../data/data.h5')
    parser.add_argument('-logfile',     dest='logfile',         default = '')
    parser.add_argument('--nocrtbgrs',  dest='create_bigrams',  action='store_false')
    parser.add_argument('--crtbgrs',    dest='create_bigrams',  action='store_true')
    parser.set_defaults(create_bigrams=False)


    return parser
