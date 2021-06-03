# Knowledge-Driven HGCL

Knowledge-Driven **H**eterogeneous **G**raph **C**ontrastive **L**earning  
This framework applies contrastive learning between two views of the input graph. To generate the two views, the framework leverages knowledge priors in the form of meta-paths to guide the transformations to focus on specific substructures within the graph.

## Pre-Training

Pre-train a Graph Convolutional Network (GCN) with specific transformation pairs

```
python pretrain.py --filepath ../data/IMDB/IMDB_processed/ --save checkpoints/IMDB/gcn.pkl --aug1 dropN_metapath --aug2 dropE --metapath 'movie,director,movie'
```

Pre-Train on all transformation pairs

```
python pretrain_all.py --filepath ../data/IMDB/IMDB_processed/ --save checkpoints/IMDB/gcn.pkl --metapath 'movie,director,movie'
```

**Note**: The files will be saved as "checkpoints/IMDB/gcn_a1_{aug1}\_a2_{aug2}.pkl" for the above examples

## Finetuning

Finetune a pre-trained GCN on a downstream task for specific transformation pair

```
python finetune.py --filepath ../data/IMDB/IMDB_processed/ --load checkpoints/IMDB/gcn_a1_dropN_not_on_metapath_a2_maskN.pkl
```

Finetune a pre-trained GCN on a downstream task for all transformation pair

```
python finetune_all.py --filepath ../data/IMDB/IMDB_processed/ --load checkpoints/IMDB/gcn.pkl
```
