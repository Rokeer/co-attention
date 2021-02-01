# Co-Attention
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

1. Run *python2 data/preprocess_asap.py* for data splitting.
2. Download GloVe pretrained embedding from: *https://nlp.stanford.edu/projects/glove*
3. Extract *glove.6B.50d.txt* to the glove folder
4. Run *python3 attn_network.py [options]* for training and evaluation

## Running on Windows

To run on Windows, do all of the commands for Linux/MacOS. Then you'll need to remove two "\n" symbols from the preprocessing script.

1. open *data/preprocess_asap.py* in your preferred text editor
2. on lines 28 and 31 in the preprocessing script, you'll find: *f_write.write("\r\n")*
3. remove the *\n* from both lines

## Results

After preprocessing the data, the program will start the training process. At the end of each epoch, the logger will
output the development and testing set scores. The highest will be kept and outputted after all epochs are complete. You can toggle the which
task (the default is ASAP3), the number of epochs (the default is 50), and more by looking at the arguments in lines 21-51 of *attn_network.py*.

Additionally, if you want to look at specific essays with their predicted and actual scores:

1. go to the *checkpoints* folder
2. after training, there should be a text file with one number per line. the line number corresponds to the essay number in the test data.
3. in the various fold directories, open the test.tsv file and compare with the predicted scored in from step 2.

## Coming Soon

1. making specific essays, as well as their predicted scores and real scores, more accessible. 
	- likely a Python script
2. updating *the preprocessing_asap.py* to be compatible with Python 3

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