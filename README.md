# Pearl_Code

These are the code files supporting the findings in paper "Decoding Nacre: Uncovering Unique Identifiers of Pearl Provenance", with co-leading authors Dahyun Kyung and Ruiqi Zhang. 


## Content List:
1. Iso_Gridsearch.ipynb: A script that uses a grid search method to find the optimal set of parameters for an SVM classifier for stable isotope data of saltwater pearls.
2. Iso_SVM.ipynb: A script that takes the optimized parameters found in gridsearch.ipynb to build an SVM classifier for saltwater pearl stable isotope data.
3. Data_Process.py: Script that pre-process all the Raman Spectrum data to match a consistent data interval, data starting/end wavenumber, and number of datapoints.
4. Raman_SVM.py: Python file that randomly selects the training dataset and the testing dataset out of the whole dataset. A developed SVM model is trained to predict the testing data.
