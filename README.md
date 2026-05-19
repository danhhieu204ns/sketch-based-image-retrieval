# Sketch based Image retrieval
## Installation

```bash
pip install -r requirements.txt
```

## Run search app

```bash
uvicorn app_sketch_search:app --reload
```

Then open `http://127.0.0.1:8000`.

## Train

### Pretrained ViT backbone

ViT-B_16

### Hyperparameters
Here is a list of full options for the model:
```bash
# dataset
data_path,            # path to load datasets.
dataset,              # choose a dataset for train or eval.
test_class,           # choose a zero-shot split of dataset.
train_split,          # train filelist to use: train or train90.

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
python scripts/split_train_9_1.py --data_path ./datasets --dataset all --seed 2021
```

This creates:

```text
*_train90.txt   # 90% of the original train filelist, used for training
*_val10.txt     # remaining 10%, kept as a held-out split
```

Then pass `--train_split train90` when training. The original zero-shot test split is unchanged.

```bash
python -u train.py --train_split train90
```

Train model on Sketchy Ext.
```bash
python -u train.py --data_path ./datasets 
                   --dataset sketchy_extend 
                   --test_class test_class_sketchy25 
                   --train_split train90 
                   --batch 15 
                   --epoch 3
                   -s ./checkpoints/sketchy_ext 
                   -c 0 
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
python -u test.py --data_path ./datasets 
  --dataset sketchy_extend 
  --test_class test_class_sketchy25 
  -l ./checkpoints/sketchy_ext/best_checkpoint.pth 
  -c 0 
  -r rn 
  --testall
```
