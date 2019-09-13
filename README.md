# co-attention
Code for BEA 13 paper "Co-Attention Based Neural Network for Source-Dependent Essay Scoring"

## Dependencies

python 2 for data/preprocess_asap.py (will be upgraded to python 3)

python 3 for the rest

* tensorflow 2.0.0 beta
* gensim
* nltk
* sklearn

run python2 data/preprocess_asap.py for data splitting.
Download Glove pretrained embedding from https://nlp.stanford.edu/projects/glove
Extract glove.6B.50d.txt to the glove folder
run python3 attn_network.py [options] for training and evaluation


## Cite
If you use the code, please cite the following paper:
```
@inproceedings{zhang2018co,
  title={Co-Attention Based Neural Network for Source-Dependent Essay Scoring},
  author={Zhang, Haoran and Litman, Diane},
  booktitle={Proceedings of the Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications},
  pages={399--409},
  year={2018}
}
```