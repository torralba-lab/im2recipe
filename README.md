# im2recipe: Learning Cross-modal Embeddings for Cooking Recipes and Food Images

This repository contains the code to train and evaluate models from the paper:  
_Learning Cross-modal Embeddings for Cooking Recipes and Food Images_

Clone it using:

```shell
git clone --recursive https://github.com/torralba-lab/im2recipe.git
```

If you find this code useful, please consider citing:

```
@inproceedings{salvador2017learning,
  title={Learning Cross-modal Embeddings for Cooking Recipes and Food Images},
  author={Salvador, Amaia and Hynes, Nicholas and Aytar, Yusuf and Marin, Javier and 
          Ofli, Ferda and Weber, Ingmar and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

## Contents
1. [Installation](#installation)
2. [Recipe1M Dataset](#recipe1m-dataset)
3. [Vision models](#vision-models)
4. [Out-of-the-box training](#out-of-the-box-training)
5. [Prepare training data](#prepare-training-data)
6. [Training](#training)
7. [Testing](#testing)
8. [Visualization](#visualization)
9. [Pretrained model](#pretrained-model)
10. [Contact](#contact)

## Installation

Install [Torch](http://torch.ch/docs/getting-started.html):
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```

Install the following packages:

```
luarocks install torch
luarocks install nn
luarocks install image
luarocks install optim
luarocks install rnn
luarocks install loadcaffe
luarocks install moonscript
```

Install CUDA and cudnn. Then run:

```
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

A custom fork of torch-hdf5 with string support is needed:

```
cd ~/torch/extra
git clone https://github.com/nhynes/torch-hdf5.git
cd torch-hdf5
git checkout chars2
luarocks build hdf5-0-0.rockspec
```

We use Python2.7 for data processing. Install dependencies with ```pip install -r requirements.txt```

## Recipe1M Dataset

Our Recipe1M dataset is available for download [here](http://im2recipe.csail.mit.edu/dataset/download).

## Vision models

We used the following pretrained vision models:

- VGG-16 ([prototxt](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt) and [caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)).

when training, point arguments ```-proto``` and ```-caffemodel``` to the files you just downloaded.

- ResNet-50 ([torchfile](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7)).

when training, point the argument ```-resnet_model``` to this file.

## Out-of-the-box training

To train the model, you will need the following files:
* `data/data.h5`: HDF5 file containing skip-instructions vectors, ingredient ids, categories and preprocessed images.
* `data/text/vocab.bin`: ingredient Word2Vec vocabulary. Used during training to select word2vec vectors given ingredient ids.

The links to download them are available [here](http://im2recipe.csail.mit.edu/dataset/download).

## Prepare training data

We also provide the steps to format and prepare Recipe1M data for training the trijoint model. We hope these instructions will allow others to train similar models with other data sources as well.

### Choosing semantic categories

We provide the script we used to extract semantic categories from bigrams in recipe titles:

- Run ```python bigrams --crtbgrs```. This will save to disk all bigrams in the corpus of all recipe titles in the training set, sorted by frequency.
- Running the same script with ```--nocrtbgrs``` will create class labels from those bigrams adding food101 categories.

These steps will create a file called ```classes1M.pkl``` in ```./data/``` that will be used later to create the HDF5 file including categories.

### Word2Vec

Training word2vec with recipe data:

- Run ```python tokenize_instructions.py train``` to create a single file with all training recipe text.
- Run the same ```python tokenize_instructions.py``` to generate the same file with data for all partitions (needed for skip-thoughts later).
- Download and compile [word2vec](https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip)
- Train with:

```
./word2vec -hs 1 -negative 0 -window 10 -cbow 0 -iter 10 -size 300 -binary 1 -min-count 10 -threads 20 -train tokenized_instructions_train.txt -output vocab.bin
```

- Run ```python get_vocab.py vocab.bin``` to extract dictionary entries from the w2v binary file. This script will save ```vocab.txt```, which will be used to create the dataset later.
- Move ```vocab.bin``` and ```vocab.txt``` to ```./data/text/```.

### Skip-instructions

- Navigate to ```th-skip```
- Create directories where data will be stored:
```
mkdir data
mkdir snaps
```

- Prepare the dataset running from ```scripts``` directory:

```
python mk_dataset.py 
--dataset /path/to/recipe1M/ 
--vocab /path/to/w2v/vocab.txt 
--toks /path/to/tokenized_instructions.txt
```

where ```tokenized_instructions.txt``` contains text instructions for the entire dataset (generated in step 2 of the Word2Vec section above), and ```vocab.txt``` are the entries of the word2vec dictionary (generated in step 6 in the previous section).


- Train the model with:

```
moon main.moon 
-dataset data/dataset.h5 
-dim 1024 
-nEncRNNs 2 
-snapfile snaps/snapfile 
-savefreq 500 
-batchSize 128 
-w2v /path/to/w2v/vocab.bin
```

- Get encoder from the trained model. From ```scripts```:

```
moon extract_encoder.moon
../snaps/snapfile_xx.t7
encoder.t7
true
```
- Extract features. From ```scripts```:

```
moon encode.moon 
-data ../data/dataset.h5
-model encoder.t7
-partition test
-out encs_test_1024.t7
```

Run for ```-partition = {train,val,test}``` and ```-out={encs_train_1024,encs_val_1024,encs_test_1024}``` to extract features for the dataset.

- Move files ```encs_*_1024.t7``` containing skip-instructions features to ```./data/text```.


### Creating HDF5 file

Navigate back to ```./```. Run the following from ```./pyscripts```:

```
python mk_dataset.py 
-vocab /path/to/w2v/vocab.txt 
-dataset /path/to/recipe1M/ 
-h5_data /path/to/h5/outfile/data.h5
-stvecs /path/to/skip-instr_files/
```

## Training

- Train the model with: 
```
th main.lua 
-dataset /path/to/h5/file/data.h5 
-ingrW2V /path/to/w2v/vocab.bin
-net resnet 
-resnet_model /path/to/resnet/model/resnet-50.t7
-snapfile snaps/snap
-dispfreq 1000
-valfreq 10000
```

*Note: Again, this can be run without arguments with default parameters if files are in the default location.*

- You can use multiple GPUs to train the model with the ```-ngpus``` flag. With 4 GTX Titan X you can set ```-batchSize``` to ~150. This is the default config, which will make the model converge in about 3 days.
- Plot loss curves anytime with ```python plotcurve.py -logfile /path/to/logfile.txt```. If ```dispfreq``` and ```valfreq``` are different than default, they need to be passed as arguments to this script for the curves to be correctly displayed. Running this script will also give you the elapsed training time. ```logifle.txt``` should contain the stdout of ```main.lua```. Redirect it with ```th main.lua > /path/to/logfile.txt ```.

## Testing

- Extract features from test set ```th main.lua -test 1 -loadsnap snaps/snap_xx.dat```. They will be saved in ```results```.
- After feature extraction, compute MedR and recall scores with ```python rank.py```.
- Extracting embeddings for any dataset partition is possible with the ```extract``` flag, which can be either ```train```, ```val``` or ```test``` (default).

## Visualization

We provide a script to visualize top-1 im2recipe examples in ```./pyscripts/vis.py ```. It will save figures under ```./data/figs/```.

## Pretrained model

Our best model can be downloaded [here](http://data.csail.mit.edu/im2recipe/im2recipe_model.t7.gz).
You can test it with:
```
th main.lua -test 1 -loadsnap im2recipe_model.t7
```

## Contact

For any questions or suggestions you can use the issues section or reach us at amaia.salvador@upc.edu or nhynes@mit.edu.
