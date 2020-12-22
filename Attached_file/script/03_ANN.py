import tensorflow
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

#Random seeds
import random as rn
def seed_everything(seed=12):
    rn.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    session_conf = tensorflow.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tensorflow.compat.v1.Session(graph=tensorflow.compat.v1.get_default_graph(), config=session_conf)
    tensorflow.compat.v1.keras.backend.set_session(sess)
seed_everything(12)

#Setting directly
os.chdir('/Users/koyama/Dropbox/Study_recently/論文/Spinach_sensory_evaluation/Spinach_paper_1/Attached_file/')
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

#Class name
y_train_ca =y_train-1
y_train_ca =to_categorical(y_train_ca, num_classes=4)
y_test_ca = y_test-1
#Import collections
class_weight ={0:1, 1:0.46, 2:0.38, 3:3.6}
model = Sequential()
# Use scikit-learn to grid search the batch size and epochs
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(learn_rate=0.01,activation='relu',dropout_rate=0.1):
#Create model
    model = Sequential()
    model.add(Dense(32, input_dim=X_test.shape[1], activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    optimizer = Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
#Create model
model = KerasClassifier(build_fn=create_model, verbose=0)
#Define the grid search parameters
batch_size = [30, 60, 90, 120]
epochs = [60, 80, 120, 150]
learn_rate = [0.001]
dropout_rate = [0.1]
activation = ['sigmoid','relu']
param_grid = dict(batch_size=batch_size, epochs=epochs,learn_rate=learn_rate, activation=activation, dropout_rate=dropout_rate)

grid = GridSearchCV(estimator=model, param_grid=param_grid ,n_jobs=-1, cv=5)
grid_result = grid.fit(scale_train, y_train_ca, verbose=1, class_weight=class_weight)
#Summarize results
print("Best: %.2f%% using %s" % (grid_result.best_score_, grid_result.best_params_))
y_pred_train = grid_result.predict(scale_train)
y_pred_test = grid_result.predict(scale_test)  
train_score = accuracy_score(y_train.values-1,y_pred_train)
test_score = accuracy_score(y_test.values-1,y_pred_test)
print("\n%s: %.2f%%" % ('val_acc', grid_result.best_score_*100))
print('train_score: '+str("{0:.2f}".format(train_score)))
print('test_score: '+str("{0:.2f}".format(test_score)))
print("train_cm:\n", confusion_matrix(y_train.values-1, y_pred_train))
print("test_cm:\n", confusion_matrix(y_test.values-1, y_pred_test))


