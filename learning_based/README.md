# Learning-Based HGCL

Learning-Based **H**eterogeneous **G**raph **C**ontrastive **L**earning  
This framework applies contrastive learning between two correlated nodes of the input graph. To generate the positve sample nodes, the framework leverages attention mechanism to generate the positve sampling distribution to sample the positve nodes.

## Semi-supervised learning setting: pre-training + finetuing + testing

Pre-train a Heterogeneous Graph Attention Network (HAN) and finetune the model, finally test the model on the unseen testing set.

ACM dataset
```
python run_ACM.py
```

DBLP dataset
```
python run_DBLP.py
```

IMDB dataset
```
python run_IMDB.py
```

**Note**: The files will be saved as "checkpoints" folder

## Unsupervised learning setting: pre-training + SVM training + SVM testing

Pre-train a Heterogeneous Graph Attention Network (HAN), train a SVM classifier using a small portion (0.2) of generated embeddings and test the model on the unseen testing embeddings set (portion: 0.8).

ACM dataset
```
python run_ACM_unsupervised.py
```

DBLP dataset
```
python run_DBLP_unsupervised.py
```

IMDB dataset
```
python run_IMDB_unsupervised.py
```
