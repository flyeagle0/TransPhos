# Source code of TransPhos


## Requirement

numpy=1.19.2
pandas=1.2.4
tensorflow=2.5.1
keras=2.5.0
backend=tensorflow
python=3.8.5

## How to use the TransPhos 

### A small amount of data can be predicted through our website 
We are glad that you can use our tool to predict protein phosphorylation sites, and here we describe in detail how to use the tool. If you only have a small number of general protein phosphorylation sites to be predicted, then we recommend you to visit our team developed website http://moleprotein.aita.ltd:8989/submittransphos  We provide a small number of computational resources on this site to perform a small number of protein phosphorylation site prediction tasks, just enter the name of your protein, the location to be predicted, and the protein sequence.

### Large amounts of data need to be predicted
If you have a large amount of protein data to be predicted, we recommend deploying your own environment to install our tools for your research. You can download all the code as a package and run the predict.py file, or if you want to train the model yourself, you can run the train.py file to train and save your own model file. We describe in more detail below about the specific inputs to these two files.


### run predict.py
The input to this file is a CSV file with one sample per row. A sample consists of three data, which are the protein name, the position of the phosphorylation site to be predicted in the protein sequence, and the complete protein sequence. If a protein has more than one position to be predicted, then each position should be used as an input sample. The following is an example；
protein_name,14,XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

**Note that the sequence here is a random sequence and not all letters represent an amino acid, but in the real input sequence you should ensure that all letters are valid amino acid symbols.Remember to specify which amino acid residue you want to predict when running, for example at the S site.**

### run train.py

If you want to train your own model, you need to run the train.py file, the input of which is also a CSV file with one sample per row, including four pieces of data information, the protein name, the position of the predicted phosphorylation site in the protein sequence, the complete protein sequence and the label of whether the position is a phosphorylation site or not, 0 means the position is not a phosphorylation site, 1 means the position position is a phosphorylation site. The following is an example.
protein_name,24,XXXXXXXXXXXXXXXXXXXXXXXXX,0
