
#####################

'''
PLEASE MAKE SURE TO RUN THIS PROJECT ON KAGGLE
EACH CELL HAS BEEN DIVIDED LIKE THE ORIGINAL WAY IT WAS PLEASE COPY PASTE THIS INTO THE MOA KAGGLE PROJECT TO THE ACCESS THE DATA
 
 
'''
######################

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold as KFold

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
test_features['cp_dose'] = np.where(test_features['cp_dose'] == 'D1', 1, 0)
test_features['cp_type'] = np.where(test_features['cp_type'] == 'trt_cp', 1, 0)
del test_features['sig_id']
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
del train_features['sig_id']
train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

del train_targets_nonscored['sig_id']
del train_targets_scored['sig_id']

train_features['cp_dose'] = np.where(train_features['cp_dose'] == 'D1', 1, 0)
train_features['cp_type'] = np.where(train_features['cp_type'] == 'trt_cp', 1, 0)

def logloss(actual, predict):
    metrics = list()
    for i in range(predict.shape[0]):
        metrics.append(log_loss(actual.values[:, i], predict[:, i]))
    return np.mean(metrics)

targets = train_targets_scored.columns

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_features['cp_dose'] = np.where(train_features['cp_dose'] == 'D1', 1, 0)
train_features['cp_type'] = np.where(train_features['cp_type'] == 'trt_cp', 1, 0)
g_features = train_features.iloc[:, 4:776]
c_features = train_features.iloc[:, 776:]

#%%

y = train_targets_scored.idxmax(axis=1)
from xgboost import XGBClassifier as xgbc
m = xgbc(tree_method='gpu_hist' )
m.fit(train_features, y)

#%%

df_feature_importance = pd.DataFrame(m.feature_importances_, index=train_features.columns, columns=['feature importance']).sort_values('feature importance', ascending=False)
#print(df_feature_importance)
imp_feats = df_feature_importance.loc[df_feature_importance['feature importance'] > 0.00001]
imp_feats.to_csv('imp_feats.csv', index=True)
print(len(imp_feats))

#%%
print(imp_feats.index)
top_feats = [train_features.columns.get_loc(c) for c in imp_feats.index if c in train_features.columns]
test = test_features.iloc[:,top_feats]


#%%

#This is interesting. I decreased the number of epochs and the validation loss has lowered. I believe this is due to reduced fitting. WeightNormalization improve results by a very small margin. Using Multilabel Stratified K Fold library as it was something we intended to use in the beginning. Goal is now to tinker with optimizer to reduce overfitting/reduce loss

fold = KFold(n_splits=4, shuffle=True)
t = train_features.iloc[:,top_feats]

test = t #train_features
temp = t.values #train_features.values
temp2 = train_targets_scored.values
samplesubmission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
samplesubmission.loc[:, train_targets_scored.columns] = 0

for index1, index2 in fold.split(test, y.values):
    Xtrain = temp[index1]
    xvalidation = temp[index2]
    Ytrain = temp2[index1]
    yvalidation = temp2[index2]

    model = tf.keras.Sequential(
        [tf.keras.layers.Input(len(t.columns)),
         #tf.keras.layers.UpSampling2D(size=2)(train_features),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.GaussianDropout(0.55),
         tfa.layers.WeightNormalization(tf.keras.layers.Dense(len(train_features.columns)*6, activation = 'relu')),
         tf.keras.layers.GaussianDropout(0.5),
         tf.keras.layers.BatchNormalization(),
         tfa.layers.WeightNormalization(tf.keras.layers.Dense(len(train_features.columns)*6, activation = 'relu')),
         tf.keras.layers.BatchNormalization(),
         tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation = 'sigmoid'))
         ])
    
    model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.RMSprop(), metrics=["accuracy"])
    model.fit(Xtrain, Ytrain, batch_size = 128, epochs = 35, callbacks = tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss', factor=0.01, patience=3, verbose=0, mode='min', min_delta=0.001, cooldown=0, min_lr=0),
              verbose = 2,validation_data=(xvalidation, yvalidation))
    predict = model.predict( test_features.iloc[:,top_feats])
    samplesubmission.loc[:, train_targets_scored.columns] += predict


#%%

samplesubmission.loc[:, train_targets_scored.columns] /= 4
samplesubmission.to_csv('submission.csv', index = False)
