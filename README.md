# Im2recipe

This repository contains the code to train and evaluate models from the paper:  
_Learning Cross-modal Embeddings for Cooking Recipes and Food Images_

Clone it using
```
git clone --recursive https://github.com/torralba-lab/im2recipe.git
```


## Contents
1. [Installation](#installation)
2. [Recipe1M Dataset](#recipe1m-dataset)
3. [Vision models](#vision-models)
4. [Out-of-the-box training](#out-of-the-box-training)
5. [Prepare training data](#prepare-training-data)
  1. [Choosing semantic categories](#choosing-semantic-categories)
  2. [Word2Vec](#word2vec)
  3. [Skip-instructions](#skip-instructions)
  4. [Creating HDF5 file](#creating-hdf5-file)
6. [Training](#training)
7. [Testing](#testing)
8. [Visualization](#visualization)
9. [More notes](#more-notes)
  1. [Training procedure](#training-procedure)
  2. [Training data](#training-data)
  3. [Image preprocessing](#image-preprocessing)
  4. [Current model and performance](#current-model-and-performance)
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

We use Python2.7 for data processing, with the following packages:
- numpy, scipy, h5py, tqdm, pillow, matplotlib, scikit-learn, word2vec, nltk and torchfile. These can be installed with:
```pip install -r requirements.txt```

## Recipe1M Dataset

Our Recipe1M dataset is available for [download](https://im2recipe.csail.mit.edu/ds_form.html).

## Vision models

The code has been tested both with VGG-16 and ResNet-50 vision models. The following files are needed:

- VGG-16 ([prototxt](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt) and [caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)).

when training, point arguments ```-proto``` and ```-caffemodel``` to the files you just downloaded.

- ResNet-50 ([torchfile](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7)).

when training, point the argument ```-resnet_model``` to this file.
  
*Note: These files are already downloaded and stored under ```./data/vision/```.*

## Out-of-the-box training

*TODO: Maybe make these files available for download:*

```
./data/data.h5 (140GB)
./data/text/vocab.bin
```

You can download these two files to start training the model right away:
- [HDF5](insert_link_here) file containing skip-instructions vectors, ingredient ids, categories and preprocessed images.
- Ingredient Word2Vec [vocabulary](insert_link_here). Used during training to select word2vec vectors given ingredient ids.


## Prepare training data

We provide the steps to format and prepare Recipe1M data for training the trijoint model. We hope these instructions will allow others to train similar models with other data sources as well.

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

Code can be found at:
``` ./th-skip/ ```

*Note: This code was cloned from [https://github.com/nhynes/th-skip](https://github.com/nhynes/th-skip) and* **slightly** *modified. Hopefully the changes I made can be added to Nick's repo and we can point to it directly. Alternatively we can just release it as a part of this repo as a fork of his.*

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

*Note: If you moved all the intermediate generated files (i.e. word2vec vocabulary and skip-thoughts vectors), you can just run ```python mk_dataset.py``` and it should work OK.*

*Note2: If we provide everything in ```./data``` (excluding .h5 files and vision models) available for download, one can skip all the steps above and simply run ```mk_dataset.py```.*

*Note3: Current version of the dataset still contains duplicates, so I use the file in ```./data/remove1M.txt``` to ignore duplicate entries. Once these are removed from the dataset (i.e from ```layerX.json``` files), the parts of the code that load and use the entries in ```remove1M.txt``` can be removed.*

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
- Plot loss curves anytime with ```python plotcurve.py -logfile /path/to/logfile.txt```. If ```dispfreq``` and ```valfreq``` are different than default, they need to be passed as arguments to this script for the curves to be correctly displayed. Running this script will also give you the elapsed training time.

## Testing

- Extract features from test set ```th main.lua -test 1 -loadsnap snaps/snap_xx.dat```. They will be saved in ```results```. 
- After feature extraction, compute MedR and recall scores with ```python rank.py```.
- Extracting embeddings for any dataset partition is possible with the ```extract``` flag, which can be either ```train```, ```val``` or ```test``` (default).

## Visualization

We provide a script to visualize top-1 im2recipe examples in ```./pyscripts/vis.py ```. It will save figures under ```./data/figs/```.

## More notes

Here I just give additional indications about the training procedure and data processing.

### Training procedure

- The model is now trained end-to-end and freezes different parts of the network as training progresses (it can be either the vision ConvNet or the rest of the model). 
- It starts freezing the vision side, until validation flattens or starts increasing.
- If validation loss does not decrease for N validations (N is specified with the ```patience``` argument), we freeze the other module of the network and continue training.
- There is also a parameter ```iter_swap``` to switch freezing after a fixed number of iterations instead of based on val loss (to use as an alternative to ```patience``` argument).
- If both ```patience=-1``` and ```iter_swap=-1``` the model will be trained all at once (no freezing). This however does not work at all for VGG-16 (tested on 800k dataset - loss fluctuates and does not decrease from 0.18) and gives worse results for ResNet-50 than when alternate freeze/unfreeze scheme is used (more on this in the note below).
- We train forever unless a maximum number of iterations are given with the ```niters``` argument. However, model snapshots are saved every time the model is validated only if it improves performance on validation. The frequency can be specified with the ```valfreq``` argument.

**Note: What I found after training several models is that cosine loss is not correlated with median rank (both computed on validation set). For instance, it was possible for me to train a model at once (no freezing) with ResNet-50 which achieved a validation loss of 0.091 - this model gave a medR of 6.8 on the validation set. The same model trained with freezing/unfreezing with ```patience=3``` for the same number of iterations & same data reached a loss of 0.103, but gives a median rank of 4.8, which is significantly better than the former.**

*Note2: When using ```patience``` parameter for freezing/unfreezing, in practice the swap only happens once (i.e. validation loss stops decreasing after the second swap, so model snapshots are no longer saved). This means that there is only one iteration: first we train embeddings, LSTMs and classifiers, then we train vision layers. This is the same procedure we described in the paper, only that now it happens automatically instead of manually stopping and finetuning the model.*

### Training data

- 1M dataset
- Images: up to 5 per recipe. This can be changed with ```maxims``` parameter when running ```./pyscripts/mk_dataset.py```.
- Instructions: Skip-thoughts (trained with torch implementation by Nick - dimension 1024)
- Ingredients: word2vec dimension 300

### Image preprocessing

- Scale so that the shortest side is 256
- Center crop of 256x256
- The above steps are applied to all images which are then saved in a h5 file. This could also be done during training from JPG files, but it makes training slower.

During training & testing:

- A random image out of the (maximum 5) images from a recipe is selected.
- Random crop of 224x224 out of 256x256 center crop.
- Horizontal flipping with 0.5 probability.

Different architectures:

- ResNet50: Input image in range 0-1, normalized with mean and std (following [this](https://github.com/facebook/fb.resnet.torch) repository, from which the ResNet-50 model was taken from).
- VGG-16: Input image in range 0-255, mean normalized.

### Current model and performance

The [best model](http://data.csail.mit.edu/im2recipe/model_resnet.zip) achieves the following performance on the test set:

| MedR1k        | Recall@1           | Recall@5  | Recall@10 |
|:-------------:|:------------------:|:---------:|:---------:|
| 4.8           | 0.2502             | 0.5259    |     0.6472|

The hyperparameters with which this model was trained are set as the default ones in the current version of the code. These are:
- Learning rate: 0.0001
- Optimizer: Adam
- Embedding dimension: 1024
- Vision Convnet: ResNet-50
- Loss weights: 0.98 --> cosine similarity, 0.01 --> recipe classification, 0.01 --> image classification
- Number of semantic categories for regularization: 1048 (1047 + 1 for background)
- Number of iterations: 230k
- Final cosine validation loss: 0.103
- Batch size: 150
- Training time (4 GTX Titan X GPUs): 3 days
- Patience: 3
- Validation frequency: 10k iterations

## Contact

These instructions were written by Amaia Salvador in December 2016. For any questions you can reach her at amaia.salvador@upc.edu. For questions about skip-thoughts torch implementation please reach Nick Hynes at nhynes@mit.edu.
