# Heterogeneous Graph Contrastive Learning (HGCL)

This framework applies contrastive learning on Heterogeneous graphs. We adopt two methods: knowledge-driven and learning-based. Knowledge-driven uses prior knowledge in the form of meta-paths to produce two views of the graph through transformations. Then it applies contrastive learning between the pairs of nodes between the two views. Learning-based uses the attention-mechanism to extract positive sample pairs and applies contrastive learning between those pairs.

## Environment Setup

Create a conda environment from the requirements.yml file

```
conda env create -n <name_of_env> -f requirements.yml
```
