# Sketch based Image retrieval
## Datasets
Please download SBIR datasets from the official websites or Google Drive and `tar -zxvf dataset` to the corresponding directory in `./datasets`. We provide train and test splits for different datasets.

### Sketchy
[Sketchy official website](https://sketchy.eye.gatech.edu/)
[Google Drive](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view?usp=sharing).

### TU-Berlin
[TU-Berlin official website](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)
[Google Drive](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view?usp=sharing).

### QuickDraw
[QuickDraw official website](https://github.com/googlecreativelab/quickdraw-dataset)
[Google Drive](https://drive.google.com/file/d/1EZ8xWRzCi8JcKiFtciD2PwguofC785gK/view?usp=sharing).

## Installation

```bash
pip install -r requirements.txt
```

## Train

### Pretrained ViT backbone

ViT-B_16

### Haperparameters
Here is a list of full options for the model:
```bash
# dataset
data_path,            # path to load datasets.
dataset,              # choose a dataset for train or eval.
test_class,           # choose a zero-shot split of dataset.

# model
cls_number,           # class number if necessary, 100 as default.
d_model,              # feature dimension, 768 as default.
d_ff,                 # fead-forward layer dimension, 1024 as default.
head,                 # number of ca encoder head, 8 as default.
number,               # number of ca encoder layer, 1 as default.
pretrained,           # whether to use pretrained ViT model, true as default.
anchor_number,        # number of anchor in rn network, 49 as default.

# train
save, -s,             # path to save checkpoints.
batch,                # batch size, 15 as default.
epoch,                # train epoch, 30 as default.
datasetLen,           # data pair for train per epoch, 10000 as default.
learning_rate,        # learning rate, 1e-5 as default.
weight_decay,         # weight decay, 1e-2 as default.

# test
load, -l,             # path to load checkpoints.
retrieval, -r,        # test method, rn for ret-token and sa for cls-token, use rn as default.
testall,              # whether use all test data, suggesting false for train, true for test.
test_sk,              # number of sketches per loop during test, 20 as default.
test_im,              # number of images per loop during test, 20 as default.
num_workers,          # dataloader num workers, 4 as default.

# other
choose_cuda, -c,      # cuda to use, 0 as default.
seed,                 # random seed, 2021 as default.
```

### Training

```bash
python -u train.py 
```

Train model on Sketchy Ext.
```bash
python -u train.py --data_path [./datasets] \
                   --dataset sketchy_extend \ 
                   --test_class test_class_sketchy25 \ 
                   --batch 15 \ 
                   --epoch 30 \ 
                   -s [./checkpoints/sketchy_ext] \
                   -c 0 \ 
                   -r rn 
```

Train model on TU-Berlin Ext.
```bash
python -u train.py --data_path [./datasets] \
                   --dataset tu_berlin \ 
                   --test_class test_class_tuberlin30 \ 
                   --batch 15 \ 
                   --epoch 30 \ 
                   -s [./checkpoints/tuberlin_ext] \
                   -c 0 \ 
                   -r rn \ 
```

Train model on QuickDraw Ext.
```bash
python -u train.py --data_path [./datasets] \
                   --dataset Quickdraw \ 
                   --test_class Quickdraw \ 
                   --batch 15 \ 
                   --epoch 30 \ 
                   -s [./checkpoints/quickdraw_ext] \
                   -c 0 \ 
                   -r rn 
```

## Evaluation

### Evaluate

```bash
python -u test.py -r rn --testall
python -u test.py -r sa --testall
```

Evaluate model on Sketchy Ext.
```bash
python -u test.py --data_path [./datasets] \
                  --dataset sketchy_extend \
                  --test_class test_class_sketchy25 \ 
                  -l [./checkpoints/sketchy_ext/best_checkpoint.pth] \
                  -c 0 \ 
                  -r rn \ 
                  --testall
```

Evaluate model on TU-Berlin Ext.
```bash
python -u test.py --data_path [./datasets] \
                  --dataset tu_berlin \
                  --test_class test_class_tuberlin30 \ 
                  -l [./checkpoints/tuberlin_ext/best_checkpoint.pth] \
                  -c 0 \ 
                  -r rn \ 
                  --testall
```

Evaluate model on QuickDraw Ext.
```bash
python -u test.py --data_path [./datasets] \
                  --dataset Quickdraw \
                  --test_class Quickdraw \ 
                  -l [./checkpoints/quickdraw_ext/best_checkpoint.pth] \
                  -c 0 \ 
                  -r rn \ 
                  --testall
```