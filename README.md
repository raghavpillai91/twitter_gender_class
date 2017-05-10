# gender_class

Prerequisties : 

    Python 2.7.12,
    pip

Install packages in python using :

    pip install -r requirements.txt


For training the model, First have to create the word2vec library - this is compulsory if it is a new data

    python make_word2vec_vocab.py

Then, train the model.

    python max_ensemble.py -q train

To test a given sentence

    python max_ensemble.py -t "sentence here"
    



