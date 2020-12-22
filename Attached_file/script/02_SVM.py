#Importing required packages.
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

#Setting directly
os.chdir('/Users/koyama/Dropbox (個人用)/Study_recently/論文/Spinach_sensory_evaluation/Spinach_paper_1/Attached_file/')

#Loading training and test dataset
df = pd.read_csv('data/train_feature.csv', index_col=0)
X_train = df.drop(['label'],axis=1)
y_train = df['label'].astype(int)
df = pd.read_csv('data/test_feature.csv', index_col=0)
X_test = df.drop(['label'],axis=1)
y_test = df['label'].astype(int)
print(len(X_train), len(X_test), len(y_train), len(y_test))

#Scaler
scaler = StandardScaler()
scaler.fit(X_train)
scale_train =scaler.transform(X_train)
scale_test =scaler.transform(X_test)
#pipelines
pipelines = {
        "svc_rbf": make_pipeline(SVC(class_weight='balanced',random_state=12))
}
svc_rbf_hyperparameters = {
    'svc__kernel': ['rbf'],
    'svc__C': np.arange(1,10,1),
    'svc__gamma':np.logspace(-4, 0, 5, base=10),
    "svc__random_state":[12]
}
hyperparameters = {
        "svc_rbf":svc_rbf_hyperparameters
}

# Create empty dictionary called fitted_models
fitted_models = {}
# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 5, n_jobs= -1,scoring='accuracy')    
    # Fit model on X_train, y_train
    model.fit(scale_train, y_train)    
    # Store model in fitted_models[name] 
    fitted_models[name] = model    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')
for name, model in fitted_models.items():
    print("Cross validation", "{0:.2f}".format(model.best_score_))
    pred = model.predict(scale_train)
    C = confusion_matrix(y_train, pred)
    print('Acc train:', "{0:.2f}".format(accuracy_score(y_train, pred)))
    print("Normalized cm train:\n", C.astype('float') / C.sum(axis=1)[:, np.newaxis])
for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(scale_test)
    C = confusion_matrix(y_test, pred)
    print('Acc test:', "{0:.2f}".format(accuracy_score(y_test, pred)))
    print("Normalized cm test:\n", C.astype('float') / C.sum(axis=1)[:, np.newaxis])
    print("Best Hyper Parameters:\n",model.best_params_)