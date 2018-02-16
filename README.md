# Biomedical Ontology Alignment: An Approach Based on Representation Learning
This repository contains our implementation of the ontology matching framework based on representation learning.

# License #

Apache License Version 2.0. For more information, please refer to the [license](LICENSE).

# Instructions for running: ##
* Prerequisites : 
    * Python, Project Jupyter.
    * Python Libraries: NumPy, SciPy, scikit-learn, Theano, Lasagne, Keras, NLTK, pickle, PyYAML.
    
* Preprocessed Data:
    * All the data used can be found in data/ directory. 
    * In each file, in the [data/ directory](data/), there are six different files. Specifically, there is:
        1. A __.yaml__ file that stores the configuration parameters.
        2. A file named __training_data__ contains the training data.
        3. A file name __pretrained-wikipedia-pubmed-and-PMC-w2v__ contains the pre-trained word vectors.
        4. Two files named __terms_of_ontology_1__ and __terms_of_ontology_2__ contain the terms of the ontology 1 and ontology 2, respectively.
        5. A file named __ground_truth_alignments__ contains the ground-truth alignments.
    
* Perform the different ontology matching tasks:
    * Launch the Jupyter Notebook App and execute the Notebook document [**Ontology_Matching**](Ontology_Matching.ipynb).
    
## Contact  ###
* prodromos DOT kolyvakis AT epfl DOT ch