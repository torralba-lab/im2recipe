import word2vec
import sys

'''
Usage: python get_vocab.py /path/to/vocab.bin
'''
w2v_file = sys.argv[1]
model = word2vec.load(w2v_file)

vocab =  model.vocab

print "Writing to files/vocab.txt..."
f = open('files/vocab.txt','w')
f.write("\n".join(vocab))
f.close()
