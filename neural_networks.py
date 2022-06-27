from collections import abc
from re import A
import numpy as np
import pandas as pd
import keras.models
import keras.layers
import keras.optimizers
import sklearn.metrics


def generate_fnn(optimizer, loss, metrics, dropout_rate, input_shape):
    '''
    Generates a fully connected three layer neural network.
    
    The first two layers have the same number of units as the number of features in the training set, while the third and
    final layer has only three units corresponding to the three possible classes (not DC, DC and paired DC). The first two
    layers have a 'ReLu' activation function, while the third and last layer has a 'Softmax' activation function. Between
    each of the dense layers there is a dropout layer with a dropout rate defined by the 'dropout_rate' parameter.

    The optimizer function is defined by the 'optimizer' parameter, the loss function by the 'loss' parameter and the
    metric used is defined by the 'metrics' parameter.
    '''
    
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(units=input_shape[1], activation='relu', input_shape=(input_shape[1],)))
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(units=input_shape[1], activation='relu'))
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(units=3, activation='softmax'))            # 3-way classification: not dc, dc or paired dc
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


# prepare train data
# ds_train = pd.read_csv('Datasets-Modified/dataset_3way_train.csv', index_col=False) # 3 way classification
ds_train = pd.read_csv('Datasets-Modified/dataset_2way_train.csv', index_col=False) # 2 way classification
ds_train.dropna(inplace=True) # why do we have NaN?
X_train = ds_train.drop(['label'], axis=1)
y_train = ds_train['label'].astype('int')

# prepare test data
# ds_test = pd.read_csv('Datasets-Modified/dataset_3way_test.csv', index_col=False) # 3 way classification
ds_test = pd.read_csv('Datasets-Modified/dataset_2way_test.csv', index_col=False) # 2 way classification
ds_test.dropna(inplace=True) # why do we have NaN?
X_test = ds_test.drop(['label'], axis=1)
y_test = ds_test['label'].astype('int')

# fully connected neural network specific parameters
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.SparseCategoricalCrossentropy()     # sparse categorical functions are used for multiclass classification
metrics = keras.metrics.SparseCategoricalAccuracy()     # when class labels are provided as integers
dropout_rate = 0.2

# early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, min_delta=0.0001, verbose=1)

# train fnn
fnn = generate_fnn(optimizer, loss, metrics, dropout_rate, X_train.shape)
fnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# predict fnn
predict = np.argmax(fnn.predict(X_test), axis=-1)

print("\n")
print("Accuracy for FNN model:  %.1f%%" % (sklearn.metrics.accuracy_score(y_test, predict)*100))
print("Precision for FNN model: %.1f%%" % (sklearn.metrics.precision_score(y_test, predict, average='macro')*100))
print("Recall for FNN model:    %.1f%%" % (sklearn.metrics.recall_score(y_test, predict, average='macro')*100))
print("F1 Score for FNN model:  %.1f%%" % (sklearn.metrics.f1_score(y_test, predict, average='macro')*100))
print("\n")

# accuracy: (tp + tn) / (p + n)
# precision: tp / (tp + fp)
# recall: tp / (tp + fn)
# f1: 2 tp / (2 tp + fp + fn)