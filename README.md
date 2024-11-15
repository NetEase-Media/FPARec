# Learning Positional Attention for Sequential Recommendation

This is our TensorFlow implementation for the paper:

*Learning Positional Attention for Sequential Recommendation*

Link: https://arxiv.org/abs/2407.02793

Please cite our paper if you use the code.

## Datasets

The preprocessed datasets are included in the repo (`e.g. data/ml-1m.txt`), where each line contains an `user id` and 
`item id` (starting from 1) meaning an interaction (sorted by timestamp).

## Model Training

To train our model FPARec on `ml-1m` with default hyper-parameters: 

```
bash scripts/train_fpa.sh
```

To train our model PARec on `ml-1m` with default hyper-parameters: 

```
bash scripts/train_pa.sh
```

## Acknowledgement 
Our code is developed based on [SASRec](https://github.com/kang205/SASRec/tree/master)*