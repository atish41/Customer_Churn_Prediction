# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 22:27:19 2023

@author: ATISHKUMAR
"""
'''
Objectives
I will explore the data and try to answer some questions like:

-What's the % of Churn Customers and customers that keep in with the active services?
-Is there any patterns in Churn Customers based on the gender?
-Is there any patterns/preference in Churn Customers based on the type of service provided?
-What's the most profitable service types?
-Which features and services are most profitable?
-Many more questions that will arise during the analysis'''

#import the libraries

import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#from catboost import CatBoostclassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score,classification_report

#load the dataset
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\project_list\cust_churn_predictions\WA_Fn-UseC_-Telco-Customer-Churn.csv')

#3.understanding the data
#Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

dataset.head()

dataset.shape

dataset.info()

dataset.columns.values

dataset.dtypes

#The target the we will use to guide the exploration is Churn
#4.visulizing missing values as matrix
msno.matrix(dataset);

#Using this matrix we can very quickly find the pattern of missingness in the dataset.

#From the above visualisation we can observe that it has no peculiar pattern that stands out. 
#In fact there is no missing data.

#5.data manipulation
dataset=dataset.drop(['customerID'],axis=1)
dataset.head()

#On deep analysis, we can find some indirect missingness in our data (which can be in form of blankspaces).
# Let's see that!

dataset['TotalCharges']=pd.to_numeric(dataset.TotalCharges,errors='coerce')
dataset.isnull().sum()

#Here we see that the TotalCharges has 11 missing values. Let's check this data.

dataset[np.isnan(dataset['TotalCharges'])]

#It can also be noted that the Tenure column is 0 for these entries even though the MonthlyCharges column is not empty.
#Let's see if there are any other 0 values in the tenure column.

dataset[dataset['tenure']==0].index

#There are no additional missing values in the Tenure column.
#Let's delete the rows with missing values in Tenure columns since there are only 11 rows and deleting them will not affect the data.

dataset.drop(labels=dataset[dataset['tenure']==0].index,axis=0,inplace=True)
dataset[dataset['tenure']==0].index

#To solve the problem of missing values in TotalCharges column, 
#I decided to fill it with the mean of TotalCharges values.

dataset.fillna(dataset['TotalCharges'].mean())

dataset.isnull().sum()

dataset['SeniorCitizen']=dataset['SeniorCitizen'].map({0:"No",1:"Yes"})
dataset.head()

dataset["InternetService"].describe(include=['object', 'bool'])

numerical_cols=['tenure', 'MonthlyCharges', 'TotalCharges']
dataset[numerical_cols].describe()