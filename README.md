# Description

Compared several important models for text classification problem. 

- Feature
  - Bag of Words (tf-idf)
  - Word Embedding (Pre-trained on Google News or trained on train dataset)

- Classification model
  - SVM
  - CNN
  - RNN - LSTM or GRU 

Train and test on the Reuters-21578 news dataset. Achieved reasonable classification accuracy (0.84~0.89).

# To run the program:

Use Jupyter lab or Jupyter notebook to open .ipynb.

1. wordemb-GoogleNews + CNN

2. wordemb-glove + CNN

3. wordemb_sum + SVM

   Glove twitter 100. Just sum all vectors.

4. tf-idf + SVM

   0.89 accuracy for rbf. Single label.

# Program File Structure

`data_structure.py` in `/data_structure`defines document objects and static statistic data we would use in building models and predicting class labels.

`preprocess.py` in `/data_preprocess` module is to read data from dataset, parser text data, translate them into list of document objects which has the class labels and feature vector. Then, tokenize the words and construct a list of class labels and bag of terms.

`metric.py` in `/metric defines` importance metric for feature selection.

# Workflow

1. Data pre-processing
   1. Construct document object
   2. Compute feature vector

2. Classification