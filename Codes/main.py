#Importing Libraries

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Data Collection and Processing

loan_dataset =pd.read_csv("loan_data.csv")

# statistical measures
loan_dataset.describe()
# number of missing values in each column
loan_dataset.isnull().sum()
# dropping the missing values
loan_dataset = loan_dataset.dropna()
# number of missing values in each column
loan_dataset.isnull().sum()
# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
# Dependent column values
loan_dataset['Dependents'].value_counts()
# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
# dependent values
loan_dataset['Dependents'].value_counts()

#Data Visualization



sns.countplot(x="Gender", data=loan_dataset, palette="hls")
plt.show()
plt.clf()
# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

# marital status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)




# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
                      
# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']                    

#


categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
#categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Loan_Amount_Term']

print(categorical_columns)
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
print(numerical_columns)

### Data Visualization libraries

fig,axes = plt.subplots(4,2,figsize=(20,20))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=loan_dataset,hue='Loan_Status',ax=axes[row,col])


plt.subplots_adjust(hspace=1)
plt.show()
plt.clf()

#Correlation Heatmap

import seaborn as sns
import matplotlib.pyplot as plt

# Set the style to a white background with rounded corners
sns.set_style("white")
plt.figure(figsize=(10, 7))

# Create a correlation heatmap
correlation_matrix = x.corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')

# Set the colorbar label font size
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# Set the font size for the annotation values
for _, spine in heatmap.spines.items():
    spine.set_visible(True)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=12)

# Set the title and adjust the layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Display the plot
plt.show()

#

sns.set(style="darkgrid")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: ApplicantIncome
sns.histplot(data=x, x="ApplicantIncome", kde=True, color='green', ax=axs[0])
axs[0].set_title("Applicant Income Distribution", fontsize=14)
axs[0].set_xlabel("Applicant Income", fontsize=12)
axs[0].set_ylabel("Frequency", fontsize=12)

# Plot 2: CoapplicantIncome
sns.histplot(data=x, x="CoapplicantIncome", kde=True, color='skyblue', ax=axs[1])
axs[1].set_title("Coapplicant Income Distribution", fontsize=14)
axs[1].set_xlabel("Coapplicant Income", fontsize=12)
axs[1].set_ylabel("Frequency", fontsize=12)

# Plot 3: LoanAmount
sns.histplot(data=x, x="LoanAmount", kde=True, color='orange', ax=axs[2])
axs[2].set_title("Loan Amount Distribution", fontsize=14)
axs[2].set_xlabel("Loan Amount", fontsize=12)
axs[2].set_ylabel("Frequency", fontsize=12)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
plt.clf()


#Training the model


import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification, make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

names = [
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Neural Net",
    ]

# separating the data and label
x = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = loan_dataset['Loan_Status']

X = x.copy()
y = y.copy()
X = X.astype(float)

def normalize(df,string):
  mean = df[string].mean()
  std = df[string].std()
  df[string] = 2.718**(-(df[string]-mean)/std)
  const = 1/(((2*3.142857)**(1/2))*std)
  return df[string]*const
for col,val in X.items():
  X[col] = normalize(X,col)

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    MLPClassifier(alpha=1, max_iter=1000),
]

linearly_separable = (X, Y)

datasets = [
            make_classification(n_samples = 480,n_features=11, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1),
            make_blobs(n_samples=480, centers=3, n_features=11,random_state=0),
            linearly_separable,
          ]
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, clf in zip(names, classifiers):
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name,score)
    print('___________________________')
parameters = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear']}

# parameters = {'solver': ['adam'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
# Uncomment this lines to run gridsearch algorithm
# X_train, X_test, y_train, y_test = train_test_split(linearly_separable[0], linearly_separable[1], test_size=0.2, random_state=42)
# clf = GridSearchCV(SVC(), parameters, n_jobs=-1)
# clf.fit(X_train,y_train)
# print(clf.score(X_test,y_test))
# print(clf.best_params_)


from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report,confusion_matrix
print(y.value_counts())
smote = SMOTE(k_neighbors=2)
oversample = SMOTEENN(smote = smote)
#oversample = SMOTE(k_neighbors=2)
X, y= linearly_separable[0], linearly_separable[1]
X, y = oversample.fit_resample(X, y)
print("Labels after oversampling")
print(y.value_counts())
print()
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
def classification(model, X, y):
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	print("Accuracy:", model.score(x_test, y_test) * 100)
	score = cross_val_score(model, X, y, cv=5)
	print("CV Score: ", np.mean(score)*100)
	print("CV Score: ",score)
	print(list(set(y_test)))
	print("Cconfusion Matrix : ",confusion_matrix(y_pred,y_test))
	print("Classification Report : ",classification_report(y_pred,y_test))


model = make_pipeline(StandardScaler(), SVC(C=100,gamma=0.1,kernel='linear')) #Best Hyperparameter Given by grid search
classification(model, X, y)

