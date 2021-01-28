# co-attention
Code for BEA 13 paper "Co-Attention Based Neural Network for Source-Dependent Essay Scoring"

## Dependencies

Python 2 for data/preprocess_asap.py (will be upgraded to Python 3).
	* I recommend that on installation, you *do not add to your PATH variable*. This way, it doesn't interfere with your current Python workflow.
	* Then, when you need to run the preprocessing script, you'll run it something like:
		* *c:/Python27/python.exe preprocess_asap.py*	

Python 3 for the rest

* tensorflow 2.0.0 beta
* gensim
	* gensim may have more dependencies, such as VS tools
* nltk
* sklearn

## Running on Linux, MacOS

1. Run python2 data/preprocess_asap.py for data splitting.
2. Download GloVe pretrained embedding from: *https://nlp.stanford.edu/projects/glove*
3. Extract *glove.6B.50d.txt* to the glove folder
4. Run *python3 attn_network.py [options]* for training and evaluation

## Running on Windows

To run on Windows, do all of the commands for Linux/MacOS. Then you'll need to remove two "\n" symbols from the preprocessing script.
	1. open "data/preprocess_asap.py" in your preferred text editor
	2. on lines 28 and 31 in the preprocessing script, you'll find: *f_write.write("\r\n")*
	3. remove the *\n* from both lines

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