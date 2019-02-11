**Requirements**
=============

1. python (2.7)
2. Python packages:
	a. matplotlib
	b. scikit-learn
	c. pandas


**Usage**
======

$ make

if Make is not installed
------------------------
$ python main.py


**Notes**
=====

1. Put the data file  "ozan_p_pApply_intern_challenge_03_20_min.csv" inside a data folder
2. Warnings will be shown by maxplotlib while showing classification result saying "Precision and F-score are ill-defined",
   because the model sometime can only classify majority class as the data is unbalanced.
   
**Description**
This is for predicting the job apply rate for user visiting Glassdoor and performing a job search. We use several machine learning algorithm to do the predicition.
In order to represent a cross-section of the most common machine learning techniques we will use following
approaches for experimentation.
• Logistic Regression
• Artificial Neural Network
• Naïve Bayes
• Decision Tree
• Random Forest
• K-Nearest Neighbor (KNN)
• Bagging (Decision Tree)
• Ada Boosting

The results of the prediction are shown in reports.pdf files.
