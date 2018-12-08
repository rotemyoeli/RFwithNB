# coding: utf-8
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
import os
import time

# Get .txt file
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("csv","*.csv"),("Any text file","*.*")])
s = file_path
os.path.split(s)
file_name = os.path.split(s)[1]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Read the data
print('DATABASE NAME - ', file_name)
df = pd.read_csv('data/' + file_name) 

print("class dataset size:")
print(df.shape)

# remove useless data
# count the number of missing elements (NaN) in each column
counter_nan = df.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]
#remove the columns with missing elements
df = df[counter_without_nan.keys()]

# Fill all missing values with the mean of the particular column
df.fillna(df.mean())

# replace values with num
df_transform = pd.get_dummies(df)

# get features (x) and scale the features
# get x and convert it to numpy array
x = df_transform.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)


# get class labels y and then encode it into number
# get class label data
y = df.ix[:,-1].values
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Validation set size:", X_test.shape)
print("Total Dataset size:", df.shape)

n_est = 5
df_res = pd.DataFrame(columns=['db_name', 'n_est', 'nonb', 'rt', 'Accu', 'MSE', 'RMSE'])

for x in range (0, 10):
    #####Run without NB
    print('##################---RF without NB---######################')
    start_time = time.time()
    clf = RandomForestClassifier (n_estimators=n_est,
                                  criterion='gini',
                                  max_depth=3,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features='auto',
                                  max_leaf_nodes=None,
                                  bootstrap=True,
                                  oob_score=False,
                                  n_jobs=1,
                                  random_state=None,
                                  verbose=0,
                                  warm_start=False,
                                  class_weight=None)
    clf = clf.fit(X_train, y_train)
    nonb_scores = clf.predict(X_test, use_nb=False)

    elapsed_time = time.time() - start_time
    print('Run time: (include fit and predict) : ', elapsed_time)
    print('RF without NB Accuracy is - ', accuracy_score(y_test, nonb_scores))

    rmseErr = mean_squared_error(nonb_scores, y_test)
    print("MSE : ")
    print(rmseErr)
    rmseErr1 = mean_squared_error(nonb_scores, y_test)**0.5
    print("RMSE : ")
    print(rmseErr1)

    # add results into panda dataframe
    df_res = df_res.append({'db_name': file_name, 'n_est': n_est, 'nonb': 1, 'rt': elapsed_time, 'Accu': (
        accuracy_score(y_test, nonb_scores)), 'MSE': rmseErr, 'RMSE': rmseErr1}, ignore_index = True)

    #####Run with NB
    print('#################---RF with NB---##################')
    start_time = time.time()

    nb_scores = clf.predict(X_test, use_nb=True)

    rmseErr = mean_squared_error(nb_scores, y_test)
    print("MSE : ")
    print(rmseErr)
    rmseErr1 = mean_squared_error(nb_scores, y_test) ** 0.5
    print("RMSE : ")
    print(rmseErr1)

    elapsed_time = time.time() - start_time
    print('Run time: (include fit and predict) : ', elapsed_time)
    print('RF with NB Accuracy is - ', accuracy_score(y_test, nb_scores))

    #add results into panda dataframe
    df_res = df_res.append({'db_name': file_name, 'n_est': n_est, 'nonb': 0, 'rt': elapsed_time, 'Accu': (
        accuracy_score(y_test, nb_scores)), 'MSE': rmseErr, 'RMSE': rmseErr1}, ignore_index=True)

    n_est = n_est + 5

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('Results.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df_res.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

plt.scatter(df_res['n_est'], df_res['Accu'])
plt.show()
print('Done')
