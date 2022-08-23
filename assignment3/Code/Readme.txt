For K means and GMM :
1)for synthetic data , put the training data and development data in the same folder as the code 
2)for the image data , change the path in the code line no 397 accordingly 

For DTW and HMM:
1)In the same folder as py files, create a folder Digits
which contain folders 1,2,3 etc.. the files of digits.
2)Also create a folder TeluguLetters which contains folder of 
Telugu letters(ai,bA,chA etc..)
3)Also extract all the files in HMM-code folder to the same folder
as py files
4)DTW.py is for performing DTW on spoken digits data.
5)DTW2.py is for performing DTW on handwritten digits data.
6)HMM.py is for performing HMM on spoken digits data
7)HMM2.py is for performing HMM on handwritten data
8)topK is a list in DTW py files that contains average of best k distances
that are considered.
9)In HMM.py, line 95 can be modified to get different
number of states and symbols.
10)Similarly line 111 in HMM2.py.
11)Letter list is in line 114 in HMM2.py and line 169 in DTW2.py
12)Digit list is in line 140 in DTW.py and line 8 in HMM.py

