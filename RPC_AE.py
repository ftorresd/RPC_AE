
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import random



# %matplotlib inline
hole = -100.0
# maxHoles = 2 # Endcap
maxHoles = 5 # Barrel

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["GOOD", "BAD"]

# df = pd.read_csv("../credit_card/data/creditcard.csv")

df2016 = pd.read_json("rpcData/occupancyBarrel2016.json")
df2017 = pd.read_json("rpcData/occupancyBarrel2017.json")
# df2016 = pd.read_json("rpcData/occupancyEndcap2016.json")
# df2017 = pd.read_json("rpcData/occupancyEndcap2017.json")
# df = df2016
df = pd.concat([df2016, df2017])


df.shape

# df.isnull().values.any()

df['Class'] = 0

count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Run class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");
# plt.savefig("count_classes.png")
plt.cla()   # Clear axis
plt.clf()   # Clear figure

bad = df[df.Class == 1]
good = df[df.Class == 0]

print "bad.shape: "+ str(bad.shape)
print "good.shape: "+ str(good.shape)


### ML STUFF ###
from sklearn import preprocessing

data = good
data = df.drop(['Class'], axis=1)

# min_max_scaler = preprocessing.MinMaxScaler()


print data
print "data.shape: "+ str(data.shape)
# data = min_max_scaler.fit_transform(data) # to be re-done. The way it is, it is losing the significance of the "0" entries.
data = data.values
temp = []
for row in data:
  rMax = np.amax(row)
  rMin = np.amin(row)
  tempRow = []
  for i in row:
    if (i == 0):
      tempRow.append(hole)
      print "HOLE!!!"
    else:
      tempRow.append( float(i-rMin)/float(rMax-rMin) )
  temp.append(tempRow)
data = pd.DataFrame(temp)

# data = preprocessing.scale(data)
# data = pd.DataFrame(data.reshape(-1, len(data)))
data = pd.DataFrame(data)

print data
print "data.shape: "+ str(data.shape)

# print data.values.reshape(-1, 1)
# data = StandardScaler().fit_transform(data.values.reshape(-1, 1))

# print data

X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
# X_train = X_train[X_train.Class == 0]
# X_train = X_train.drop(['Class'], axis=1)

X_test_GOOD, X_test_BAD = train_test_split(X_test, test_size=0.5, random_state=RANDOM_SEED)
# X_test_BAD.loc[:,('Class')] = 1
# X_test_GOOD.loc[:,('Class')] = 0
X_test_GOOD['Class'] = 0

# Toy BAD data
print "#"*200
from sklearn.utils import shuffle
print X_test_BAD
X_test_BAD = X_test_BAD.values
temp = []
for row in X_test_BAD:
  # rand = random.random()
  rand = 0 # only shuffle
  if (rand > 0.5):
    for i in range(np.random.randint(0,maxHoles)):
      row = [hole if x == random.choice(row) else x for x in row]
  else:
    # row = np.zeros(*row.shape)hole
    row = shuffle(row).tolist()
  temp.append(row)

X_test_BAD = pd.DataFrame(np.array(temp))
X_test_BAD['Class'] = 1
print X_test_BAD
print "#"*200



X_test = pd.concat([X_test_GOOD, X_test_BAD])

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values



# default model
# input_dim = X_train.shape[1]
# encoding_dim = 14
# input_layer = Input(shape=(input_dim, ))
# encoder = Dense(encoding_dim, activation="tanh", 
#                 activity_regularizer=regularizers.l1(10e-5))(input_layer)
# encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
# decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
# decoder = Dense(input_dim, activation='relu')(decoder)
# autoencoder = Model(inputs=input_layer, outputs=decoder)


input_dim = X_train.shape[1]
encoding_dim = 40
input_layer = Input(shape=(input_dim, ))
# encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim/2 ), activation="relu")(encoder)
encoder = Dense(int(encoding_dim/4 ), activation="relu")(encoder)
# encoder = Dense(int(encoding_dim/8 ), activation="relu")(input_layer)
encoder = Dense(int(encoding_dim/8 ), activation="relu")(encoder)
decoder = Dense(int(encoding_dim/4), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim/2), activation='relu')(decoder)
decoder = Dense(int(encoding_dim), activation='sigmoid')(decoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)


from keras.utils import plot_model
plot_model(autoencoder, to_file='model.png')

print(autoencoder.summary())

nb_epoch = 100
# batch_size = len(X_train)
batch_size = 5
# batch_size = 30

# autoencoder.compile(optimizer='adam', 
#                     loss='mean_squared_error', 
#                     metrics=['accuracy'])

autoencoder.compile(optimizer='adadelta', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

autoencoder = load_model('model.h5')




plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
# plt.show()
plt.savefig("loss_val_loss.png")
plt.cla()   # Clear axis
plt.clf()   # Clear figure

predictions = autoencoder.predict(X_test)


mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})

print error_df.describe()
print error_df


# fig = plt.figure()
# ax = fig.add_subplot(111)
# good_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
good_error_df = error_df[(error_df['true_class']== 0)]
# print good_error_df
# _ = ax.hist(good_error_df.reconstruction_error.values, bins=50)
# fig = plt.figure()
# ax = fig.add_subplot(111)
bad_error_df = error_df[error_df['true_class'] == 1]
# _ = ax.hist(bad_error_df.reconstruction_error.values, bins=50)
# bins = np.linspace(0, 100, 100)
binRange = [0.0, 0.4]
plt.hist(good_error_df.reconstruction_error.values, bins=100, range = binRange, alpha=0.5,  label='GOOD')
plt.hist(bad_error_df.reconstruction_error.values, bins=100, range = binRange, alpha=0.5, label='BAD')
plt.legend(loc='upper right')
plt.savefig("reconstruction_error.png")
plt.cla()   # Clear axis
plt.clf()   # Clear figure


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show();
plt.savefig("ROC.png")
plt.cla()   # Clear axis
plt.clf()   # Clear figure

# precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
# plt.plot(recall, precision, 'b', label='Precision-Recall curve')
# plt.title('Recall vs Precision')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# # plt.show()
# plt.savefig("a03.png")



# plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
# plt.title('Precision for different threshold values')
# plt.xlabel('Threshold')
# plt.ylabel('Precision')
# # plt.show()
# plt.savefig("a03.png")



# plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
# plt.title('Recall for different threshold values')
# plt.xlabel('Reconstruction error')
# plt.ylabel('Recall')
# # plt.show()
# plt.savefig("a04.png")


threshold = 0.04
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "BAD" if name == 1 else "GOOD")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
# plt.show();
plt.savefig("threshold.png")
plt.cla()   # Clear axis
plt.clf()   # Clear figure






def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout()



y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
print conf_matrix

# plt.figure(figsize=(12, 12))

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix, classes=LABELS, normalize=True,
                      title='Normalized confusion matrix')

# sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

# plt.title("Confusion matrix")
# plt.ylabel('True class')
# plt.xlabel('Predicted class')
# plt.show()
plt.savefig("confusion_matrix.png")
plt.cla()   # Clear axis
plt.clf()   # Clear figure


