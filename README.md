Final Project for CSCI-GA.3033 Big Data Science
Authors: Jiajun Bao, Meng Li, Jane Liu


FILES
main.py
dataset.py
train.py
neuralnetwork.py
preprocessor.py
train.py
validate.py
lda_knn.py
detecting_sys_window.py
fp.s
lda.s
/data/
    reviewContent.txt
    metadata.txt
    stopwords.txt
    test.csv
    train.csv
    validate.csv


REQUIREMENTS
Please copy all files and folders into the working directory (shown above). The /data/ folder is in the same directory as main.py.

The following libraries are used: nltk, numpy, pandas, statsmodels, patsy (required by statsmodels), sklearn, matplotlib, gensim. To set up the environment cd to the working directory and enter:

$ module purge
$ module load python3/intel/3.6.3
$ virtualenv venv
$ source venv/bin/activate
$ pip install -U nltk
$ pip install -U pandas     # this should automatically install numpy
$ pip install -U --no-deps statsmodels
$ pip install -U patsy      # required by statsmodels
$ pip install -U sklearn
$ pip install -U matplotlib
$ pip install -U gensim

This program was written in Python 3.6. It is unknown if it will work correctly for other versions of Python.


INSTRUCTIONS
1. There are two programs in our project. The first program runs the logistic regression and neural network models and the second program runs the LDA and knn models.

2. To run this project as a batch process please enter "sbatch fp.s" and then enter "sbatch lda.s" (in the bash files the working directory file path needs to be updated). fp.s is for the logistic regression and neural network model and lda.s is for the LDA and kNN model.

3. If running from the terminal enter the virtual environment then enter "python main.py" and "python lda_knn.py".

4. Our program for deployment (detecting_sys_window.py) allows a user to input an unknown review and obtain a result (fake or non-fake). It does not work when the project is run as a batch process on Prince. The program can be run from the terminal on a local machine with sufficient memory (simply uncomment the last two lines of code in main.py and type "python main.py").


EXPECTED RESULTS
The logistic regression model is expected to have 70% accuracy.
The neural network model is expected to have 71% accuracy.
The LDA and kNN model is reported to have 60% accuracy.

(expected output from slurm:)

Exploring the metadata.txt file:  
    userID  b  rating label        date 
0     923  0     3.0     -1  	2014-12-08 
1     924  0     3.0     -1  	2013-05-16 
2     925  0     4.0     -1  	2013-07-01 
3     926  0     4.0     -1  	2011-07-28 
4     927  0     4.0     -1  	2010-11-01 
5     928  0     4.0     -1  	2009-09-02 
6     929  0     4.0     -1  	2009-08-25 
7     930  0     4.0     -1  	2007-05-20 
8     931  0     4.0     -1  	2005-12-27 
9     932  0     5.0     -1  	2014-05-09 

               		userID              b                    rating              label             date 
count   	  359052.00  	  359052.00    	     359052.00    359052.00     359052 
unique            NaN            	NaN            	NaN              NaN         	   3417 
top                  NaN           	NaN            	NaN              NaN          2015-01-05 
freq                NaN            	NaN            	NaN              NaN          	   454 
mean     	53992.205533     459.929601       4.025871       0.794542         NaN 
std      	45806.707721     259.923732       1.055061       0.607210         NaN 
min        	923.000000         0.000000           1.000000      -1.000000         NaN 
25%      	13840.000000     247.000000       4.000000       1.000000         NaN 
50%      	40523.000000     468.000000       4.000000       1.000000         NaN 
75%      	87314.000000     672.000000       5.000000       1.000000         NaN 
max     	161147.000000   922.000000       5.000000       1.000000         NaN 
 
 
Exploring the reviewContent.txt file:  
    userID  b        date                                            content 
0     923  0  2014-12-08  The food at snack is a selection of popular Gr... 
1     924  0  2013-05-16  This little place in Soho is wonderful. I had ... 
2     925  0  2013-07-01  ordered lunch for 15 from Snack last Friday.  ... 
3     926  0  2011-07-28  This is a beautiful quaint little restaurant o... 
4     927  0  2010-11-01  Snack is great place for a  casual sit down lu... 
5     928  0  2009-09-02  A solid 4 stars for this greek food spot.  If ... 
6     929  0  2009-08-25  Let me start with a shout-out to everyone who ... 
7     930  0  2007-05-20  Love this place!  Try the Chicken sandwich or ... 
8     931  0  2005-12-27  My friend and I were intrigued by the nightly ... 
9     932  0  2014-05-09  Stopped in for lunch today and couldn't believ... 

               	    userID              	   b        	  date     	  content 
count   	 358957.00  	   358957.00     358957      358957 
unique          NaN            		NaN        	 3417          358080 
top               NaN               	NaN      2015-01-05   Delicious! 
freq              NaN               	NaN           454          20 
mean     	53997.215187    459.989378	  NaN         NaN 
std       	45808.754776	   259.913081    NaN         NaN 
min        	923.000000         0.000000       NaN         NaN 
25%      	13840.000000     247.000000   NaN         NaN 
50%      	40532.000000     468.000000   NaN         NaN 
75%      	87321.000000     672.000000   NaN         NaN 
max     	161147.000000   922.000000   NaN         NaN 

The distribution of Yelp review ratings: 
5.0    141157 
4.0    135250 
3.0     47646 
2.0     20775 
1.0     14224 
Name: rating, dtype: int64 
 
The number of reviewers (spammers are indicated by '-1' value): 
 1    322167 
-1     36885 
Name: label, dtype: int64


Optimization terminated successfully.
         Current function value: 0.588500
         Iterations 6
                           Logit Regression Results
==============================================================================
Dep. Variable:                     label   	No. Observations:               125531
Model:                                 Logit   	Df Residuals:                       125515
Method:                                MLE   	Df Model:                                    15
Date:                Sat, 18 May 2019   Pseudo R-squ.:                     0.1510
Time:                             14:36:20   	Log-Likelihood:                     -73875.
converged:                          True   	LL-Null:                                 -87011.
                                             LLR 	p-value:                                   0.000
====================================================================================
                       	      coef        std err        z      	   P>|z|        [0.025      0.975]
------------------------------------------------------------------------------------
length_of_review  7.7870      0.093     83.894    0.000       7.605       7.969
order                   0.4303      0.168      2.563      0.010       0.101       0.760
thing                   0.3251       0.142      2.282      0.022       0.046       0.604
good                   -0.1892      0.088     -2.145      0.032      -0.362      -0.016
side                    1.3556       0.139      9.723      0.000       1.082       1.629
day                     0.8866       0.164      5.419      0.000       0.566       1.207
bit                       0.9866       0.144      6.839      0.000       0.704       1.269
flavor                  1.4055       0.172      8.165      0.000       1.068       1.743
pretty                  5.0092       0.165     30.420      0.000       4.687       5.332
sauce                 1.3159       0.167      7.888      0.000       0.989       1.643
star                     1.4048       0.272      5.163      0.000       0.872       1.938
rating_2.0           1.4466       0.049     29.735      0.000       1.351       1.542
rating_3.0           1.9621       0.043     45.205      0.000       1.877       2.047
rating_4.0           1.7422       0.041     42.060      0.000       1.661       1.823
rating_5.0           1.0375       0.042     24.975      0.000       0.956       1.119
intercept           -2.4525        0.042    -58.597      0.000      -2.534      -2.370
====================================================================================
[[16434  4491]
 [ 8054 12864]]

              precision    recall  f1-score   support

           0       0.67      0.79      0.72     20925
           1       0.74      0.61      0.67     20918

    accuracy                                    0.70     41843
   macro avg       0.71      0.70       0.70     41843
weighted avg       0.71      0.70      0.70     41843


Neural Network accuracy score: 0.7076452453217982


