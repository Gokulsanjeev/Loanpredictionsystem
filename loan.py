# %% [markdown]
# # Introduction
# 
# <table>
#   <tr><td>
#     <img src="https://pas-wordpress-media.s3.us-east-1.amazonaws.com/content/uploads/2015/12/loan-e1450497559334.jpg"
#          width="400" height="600">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# In finance, a loan is the lending of money by one or more individuals, organizations, or other entities to other individuals, organizations etc. The recipient (i.e., the borrower) incurs a debt and is usually liable to pay interest on that debt until it is repaid as well as to repay the principal amount borrowed. ([wikipedia](https://en.wikipedia.org/wiki/Loan))
# 
# ### **The major aim of this notebook is to predict which of the customers will have their loan approved.**
# 
# ![](https://i.pinimg.com/originals/41/b0/08/41b008395e8e7f888666688915750d1f.gif)
# 
# # Data Id 📋
# 
# This dataset is named [Loan Prediction Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset) data set. The dataset contains a set of **613** records under **13 attributes**:
# 
# ![](http://miro.medium.com/max/795/1*cAd_tqzgCWtCVMjEasWmpQ.png)
# 
# ## The main objective for this dataset:
# Using machine learning techniques to predict loan payments.
# 
# ### target value: `Loan_Status`
# 
# # Libraries 📕📗📘

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.761878Z","iopub.status.idle":"2022-07-26T17:37:30.762641Z"}}
import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter


#ploting libraries
import matplotlib.pyplot as plt 
import seaborn as sns

#relevant ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#ML models
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#default theme
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

#warning hadle
warnings.filterwarnings("ignore")

# %% [markdown]
# # File path 📂

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.764258Z","iopub.status.idle":"2022-07-26T17:37:30.765008Z"}}
#list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.766489Z","iopub.status.idle":"2022-07-26T17:37:30.76722Z"}}
#path for the training set
tr_path = "train_u6lujuX_CVtuZ9i.csv"
#path for the testing set
te_path = "test_Y3wMUE5_7gLdaTN.csv"

# %% [markdown]
# # Preprocessing and Data Analysis 💻
# 
# ## First look at the data:
# 
# Training set:

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-07-26T17:37:30.768627Z","iopub.status.idle":"2022-07-26T17:37:30.769331Z"}}
# read in csv file as a DataFrame
tr_df = pd.read_csv(tr_path)
# explore the first 5 rows
tr_df.head()

# %% [markdown]
# Testing set:

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-07-26T17:37:30.77099Z","iopub.status.idle":"2022-07-26T17:37:30.771784Z"}}
# read in csv file as a DataFrame
te_df = pd.read_csv(te_path)
# explore the first 5 rows
te_df.head()

# %% [markdown]
# Size of each data set:

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-07-26T17:37:30.773376Z","iopub.status.idle":"2022-07-26T17:37:30.774143Z"}}
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")

# %% [markdown]
# ### Now the focus is shifted for the preprocessing of the training dataset.

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.775621Z","iopub.status.idle":"2022-07-26T17:37:30.776322Z"}}
#column information
tr_df.info(verbose=True, null_counts=True)

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.777815Z","iopub.status.idle":"2022-07-26T17:37:30.778557Z"}}
#summary statistics
tr_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.780483Z","iopub.status.idle":"2022-07-26T17:37:30.781258Z"}}
#the Id column is not needed, let's drop it for both test and train datasets
tr_df.drop('Loan_ID',axis=1,inplace=True)
te_df.drop('Loan_ID',axis=1,inplace=True)
#checking the new shapes
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")

# %% [markdown]
# ## Missing values 🚫
# As you can see we have some missing data, let's have a look how many we have for each column:

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.782735Z","iopub.status.idle":"2022-07-26T17:37:30.783481Z"}}
#missing values in decsending order
tr_df.isnull().sum().sort_values(ascending=False)

# %% [markdown]
# Each value will be replaced by the most frequent value (mode).
# 
# E.G. `Credit_History` has 50 null values and has 2 unique values `1.0` (475 times) or `0.0` (89 times) therefore each null value will be replaced by the mode `1.0` so now it will show in our data 525 times. 

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.784747Z","iopub.status.idle":"2022-07-26T17:37:30.785435Z"}}
#filling the missing data
print("Before filling missing values\n\n","#"*50,"\n")
null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']


for col in null_cols:
    print(f"{col}:\n{tr_df[col].value_counts()}\n","-"*50)
    tr_df[col] = tr_df[col].fillna(
    tr_df[col].dropna().mode().values[0] )   

    
tr_df.isnull().sum().sort_values(ascending=False)
print("After filling missing values\n\n","#"*50,"\n")
for col in null_cols:
    print(f"\n{col}:\n{tr_df[col].value_counts()}\n","-"*50)

# %% [markdown]
# ## Data visalization 📊

# %% [markdown]
# Firstly we need to split our data to categorical and numerical data,
# 
# 
# using the `.select_dtypes('dtype').columns.to_list()` combination.

# %% [markdown]
# ## Loan status distribution

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.787124Z","iopub.status.idle":"2022-07-26T17:37:30.787843Z"}}
#list of all the columns.columns
#Cols = tr_df.tolist()
#list of all the numeric columns
num = tr_df.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = tr_df.select_dtypes('object').columns.to_list()

#numeric df
loan_num =  tr_df[num]
#categoric df
loan_cat = tr_df[cat]

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.789153Z","iopub.status.idle":"2022-07-26T17:37:30.78988Z"}}
print(tr_df[cat[-1]].value_counts())
#tr_df[cat[-1]].hist(grid = False)

#print(i)
total = float(len(tr_df[cat[-1]]))
plt.figure(figsize=(8,10))
sns.set(style="whitegrid")
ax = sns.countplot(tr_df[cat[-1]])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 
plt.show()

# %% [markdown]
# Let's plot our data
# 
# Numeric:

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.791159Z","iopub.status.idle":"2022-07-26T17:37:30.791885Z"}}
for i in loan_num:
    plt.hist(loan_num[i])
    plt.title(i)
    plt.show()


# %% [markdown]
# Categorical (split by Loan status):

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.793522Z","iopub.status.idle":"2022-07-26T17:37:30.79426Z"}}
for i in cat[:-1]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='Loan_Status', data=tr_df ,palette='plasma')
    plt.xlabel(i, fontsize=14)

# %% [markdown]
# ## Encoding data to numeric

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.7957Z","iopub.status.idle":"2022-07-26T17:37:30.796413Z"}}
#converting categorical values to numbers

to_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}

# adding the new numeric values from the to_numeric variable to both datasets
tr_df = tr_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
te_df = te_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)

# convertind the Dependents column
Dependents_ = pd.to_numeric(tr_df.Dependents)
Dependents__ = pd.to_numeric(te_df.Dependents)

# dropping the previous Dependents column
tr_df.drop(['Dependents'], axis = 1, inplace = True)
te_df.drop(['Dependents'], axis = 1, inplace = True)

# concatination of the new Dependents column with both datasets
tr_df = pd.concat([tr_df, Dependents_], axis = 1)
te_df = pd.concat([te_df, Dependents__], axis = 1)

# checking the our manipulated dataset for validation
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}\n")
print(tr_df.info(), "\n\n", te_df.info())

# %% [markdown]
# ## Correlation matrix 

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.797821Z","iopub.status.idle":"2022-07-26T17:37:30.798575Z"}}
#plotting the correlation matrix
sns.heatmap(tr_df.corr() ,cmap='cubehelix_r')

# %% [markdown]
# ### Correlation table for a more detailed analysis:

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.800073Z","iopub.status.idle":"2022-07-26T17:37:30.800651Z"}}
#correlation table
corr = tr_df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# %% [markdown]
# We can clearly see that `Credit_History` has the highest correlation with `Loan_Status` (a positive correlation of `0.54`).
# Therefore our target value is highly dependant on this column.

# %% [markdown]
# # Machine learning models
# 
# First of all we will divide our dataset into two variables `X` as the features we defined earlier and `y` as the `Loan_Status` the target value we want to predict.
# 
# ## Models we will use:
# 
# * **Decision Tree** 
# * **Random Forest**
# * **XGBoost**
# * **Logistic Regression**
# 
# ## The Process of Modeling the Data:
# 
# 1. Importing the model
# 
# 2. Fitting the model
# 
# 3. Predicting Loan Status
# 
# 4. Classification report by Loan Status
# 
# 5. Overall accuracy
# 

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.802306Z","iopub.status.idle":"2022-07-26T17:37:30.802826Z"}}
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)
X,y = tr_df.drop(columns = ['Loan_Status']), tr_df['Loan_Status']
X, y = ros.fit_resample(X,y)
from collections import Counter
print(sorted(Counter(y_train).items()))


# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.804298Z","iopub.status.idle":"2022-07-26T17:37:30.804838Z"}}
X_train

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.806232Z","iopub.status.idle":"2022-07-26T17:37:30.806929Z"}}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# %% [markdown]
# ## Decision Tree
# 
# ![](https://i.pinimg.com/originals/eb/08/05/eb0805eb6e34bf3eac5ab4666bbcc167.gif)

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.808718Z","iopub.status.idle":"2022-07-26T17:37:30.809386Z"}}
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

y_predict = DT.predict(X_test)

#  prediction Summary by species
print(classification_report(y_test, y_predict))

# Accuracy score
DT_SC = accuracy_score(y_predict,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")

# %% [markdown]
# ### Csv results of the test for our model:
# 
# <table>
#   <tr><td>
#     <img src="https://miro.medium.com/max/900/1*a99bY1VkmfXhqW-5uAX28w.jpeg"
#          width="200" height="300">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# You can see each predition and true value side by side by the csv created in the output directory.

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.81085Z","iopub.status.idle":"2022-07-26T17:37:30.811343Z"}}
Decision_Tree=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Decision_Tree.to_csv("Dection Tree.csv")     

# %% [markdown]
# ## Random Forest
# 
# ![](https://miro.medium.com/max/1280/1*9kACduxnce_JdTrftM_bsA.gif)

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.812584Z","iopub.status.idle":"2022-07-26T17:37:30.813166Z"}}
RF = RandomForestClassifier()
RF.fit(X_train, y_train)

y_predict = RF.predict(X_test)

#  prediction Summary by species
print(classification_report(y_test, y_predict))

# Accuracy score
RF_SC = accuracy_score(y_predict,y_test)
print(f"{round(RF_SC*100,2)}% Accurate")

# %% [markdown]
# ### Csv results of the test for our model:
# 
# <table>
#   <tr><td>
#     <img src="https://miro.medium.com/max/900/1*a99bY1VkmfXhqW-5uAX28w.jpeg"
#          width="200" height="300">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# You can see each predition and true value side by side by the csv created in the output directory.

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.814358Z","iopub.status.idle":"2022-07-26T17:37:30.814914Z"}}
Random_Forest=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Random_Forest.to_csv("Random Forest.csv")     

# %% [markdown]
# ## XGBoost
# 
# ![](https://f-origin.hypotheses.org/wp-content/blogs.dir/253/files/2015/06/boosting-algo-3.gif)

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.816258Z","iopub.status.idle":"2022-07-26T17:37:30.816856Z"}}
XGB = XGBClassifier()
XGB.fit(X_train, y_train)

y_predict = XGB.predict(X_test)

#  prediction Summary by species
print(classification_report(y_test, y_predict))

# Accuracy score
XGB_SC = accuracy_score(y_predict,y_test)
print(f"{round(XGB_SC*100,2)}% Accurate")

# %% [markdown]
# ### Csv results of the test for our model:
# 
# <table>
#   <tr><td>
#     <img src="https://miro.medium.com/max/900/1*a99bY1VkmfXhqW-5uAX28w.jpeg"
#          width="200" height="300">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# You can see each predition and true value side by side by the csv created in the output directory.

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.817963Z","iopub.status.idle":"2022-07-26T17:37:30.818733Z"}}
XGBoost=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
XGBoost.to_csv("XGBoost.csv")     

# %% [markdown]
# ## Logistic Regression
# Now, I will explore the Logistic Regression model.
# 
# <table>
#   <tr><td>
#     <img src="https://files.realpython.com/media/log-reg-2.e88a21607ba3.png"
#           width="500" height="400">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.81976Z","iopub.status.idle":"2022-07-26T17:37:30.820299Z"}}
LR = LogisticRegression()
LR.fit(X_train, y_train)

y_predict = LR.predict(X_test)

#  prediction Summary by species
print(classification_report(y_test, y_predict))

# Accuracy score
LR_SC = accuracy_score(y_predict,y_test)
print('accuracy is',accuracy_score(y_predict,y_test))

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.821632Z","iopub.status.idle":"2022-07-26T17:37:30.822182Z"}}
Logistic_Regression=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Logistic_Regression.to_csv("Logistic Regression.csv")     

# %% [markdown]
# ### Csv results of the test for our model:
# 
# <table>
#   <tr><td>
#     <img src="https://miro.medium.com/max/900/1*a99bY1VkmfXhqW-5uAX28w.jpeg"
#          width="200" height="300">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# You can see each predition and true value side by side by the csv created in the output directory.

# %% [markdown]
# # Conclusion
# 
# 1. `Credit_History` is a very important variable  because of its high correlation with `Loan_Status` therefor showind high Dependancy for the latter.
# 2. The Logistic Regression algorithm is the most accurate: **approximately 83%**.

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.823338Z","iopub.status.idle":"2022-07-26T17:37:30.823941Z"}}
score = [DT_SC,RF_SC,XGB_SC,LR_SC]
score = [str(int(i)*100)+" %" for i in score]
Models = pd.DataFrame({
    'n_neighbors': ["Decision Tree","Random Forest","XGBoost", "Logistic Regression"],
    'Accuracy': score})
Models.sort_values(by='Accuracy', ascending=False)

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:42:38.093045Z","iopub.execute_input":"2022-07-26T17:42:38.093578Z","iopub.status.idle":"2022-07-26T17:42:38.100303Z","shell.execute_reply.started":"2022-07-26T17:42:38.093544Z","shell.execute_reply":"2022-07-26T17:42:38.098949Z"}}
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical 

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:37:30.825038Z","iopub.status.idle":"2022-07-26T17:37:30.825615Z"}}
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:58:21.675187Z","iopub.execute_input":"2022-07-26T17:58:21.675573Z","iopub.status.idle":"2022-07-26T17:58:30.76663Z","shell.execute_reply.started":"2022-07-26T17:58:21.675532Z","shell.execute_reply":"2022-07-26T17:58:30.765442Z"}}
model = Sequential()
model.add(Dense(200, activation='relu', input_dim=11))
model.add(Dense(50, activation='relu'))

model.add(Dense(80, activation='relu'))

model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, validation_split=0.3, shuffle=True,)

# %% [code] {"execution":{"iopub.status.busy":"2022-07-26T17:59:53.294591Z","iopub.execute_input":"2022-07-26T17:59:53.294955Z","iopub.status.idle":"2022-07-26T17:59:53.605654Z","shell.execute_reply.started":"2022-07-26T17:59:53.294924Z","shell.execute_reply":"2022-07-26T17:59:53.604375Z"}}
pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

# %% [markdown]
# 
# 
# 