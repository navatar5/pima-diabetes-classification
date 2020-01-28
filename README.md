# pima-diabetes-classification
This project compares compares the classification performance of Naive Bayes, SVM, and Decision Trees on the Pima Indians diabetes dataset.

The dataset can be found at: https://www.kaggle.com/uciml/pima-indians-diabetes-database/data

See the Jupyter notebook: https://github.com/navatar5/pima-diabetes-classification/blob/master/PimaDiabetesClassification.ipynb

The repository consists of the following files:

datatools.py: Contains helper functions for data loading and processing

crossvalidate.py: Contains a leave one out crossvalidation function and a n-fold crossvalidation function

NaiveBayes.py: Naive Bayes Classifier, contains train, run and helper functions

SVM.py: SVM classifier using sklearn, contains train and run functions

DTscractch.py: Decision Tree Classifier, contains train, run and helper functions

DecisionTree.py: Decision Tree Classifier using sklearn, contains train and run functions



I found the tutorials by Jason Brownlee's machinelearingmastery.com very hepful in understanding and coding the Naive Bayes and Decision Tree algorithms
"from scratch". Links below:

https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/







