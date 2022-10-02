from genericpath import exists
import numpy as np
import tensorflow as tf
import model
import os
from tensorflow import keras

from dataloader import loadCSV
from dataloader import IEGM_DataGenerator, IEGM_DataGenerator_test

from sklearn.model_selection import StratifiedKFold

os.makedirs('./trained_models', exist_ok=True)

seed = 7
np.random.seed(seed)

# set hyperparameters
K = 5
epoch = 50
bchsz = 16
LR = 0.0001
name_post = 'ds_max5'
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

# load data
partition, labels = {}, {}
partition['train'] = []
partition['test'] = []
train_csv_data = loadCSV(os.path.join('./data_indices/' + 'train_indice.csv'))
test_csv_data = loadCSV(os.path.join('./data_indices/', 'test_indice.csv'))
for k, v in train_csv_data.items():
    partition['train'].append(k)
    labels[k] = v[0]
        
for k, v in test_csv_data.items():
    partition['test'].append(k)
    labels[k] = v[0]
print("indices loaded")

train_dataset = IEGM_DataGenerator(partition['train'], labels, bchsz, shuffle=True, size=1250)
test_dataset = IEGM_DataGenerator_test(partition['test'], labels, batch_size=bchsz, shuffle=True, size=1250)
print("data loaded")
X = train_dataset[0][0]
Y_raw = train_dataset[0][1]
x_test = test_dataset[0][0]
y_test = test_dataset[0][1]
Y = np.ndarray(shape=(24588, 1), dtype=int)
for i, _ in enumerate(Y_raw):
    if Y_raw[i][0] == 1:
        Y[i] = 0 
    else:
        Y[i] = 1

# data augmentation
rng = np.random.default_rng()
factor = rng.uniform(low=0.8, high=1.2, size=( 24588, 1, 1250, 1))
X = X * factor 
print("data augmented")

# for computing scores
def count(y_predict, y_label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i, (a, b) in enumerate(y_label):
        (c, d) = y_predict[i]
        if a == 0:
            if c == 0:
                TP += 1
            else:
                FN += 1
        else:
            if c == 1:
                TN += 1
            else:
                FP += 1
    return ([TP, FN, FP, TN])
         
def convertmax(y_predict):
    y = np.empty((y_predict.shape[0]), dtype=int) 
    for i, (a,b) in enumerate(y_predict):
        if(a >= b):
            y[i] = 0
        else:
            y[i] = 1
    return keras.utils.to_categorical(y, num_classes=2)

def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        fb = 0
    else:
        fb = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return fb 

def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]

    if tp + fn == 0:
        ppv = 1
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv

def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity

def main():
    print("Epoch: ", epoch)
    print("K = ", K)
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    cvscores = []
    fbs = []
    i = 1
    for train, test in kfold.split(X, Y):
        print(f"fold {i} out of {K}:")
        # load model
        new_model = model.model_ds_max5() ###
        # new_model.summary()

        # compile model & set callback info.
        new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')
        checkpoint_filepath = './trained_models/model_%s_%d.hdf5' %(name_post, i)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1,
            save_best_only=True)

        # start training
        history = new_model.fit(
            X[train], 
            Y_raw[train], 
            validation_data=(X[test], Y_raw[test]),
            batch_size=bchsz,
            epochs=epoch,
            verbose=0,
            callbacks=[model_checkpoint_callback]
            )
    
        # load best weights
        new_model.load_weights(checkpoint_filepath)

        # evaluate the model & compute scores
        pred = new_model.predict(x_test)
        predict = convertmax(pred)
        list = count(predict, y_test)
        fb = FB(list)
        _, baseline_model_accuracy = new_model.evaluate(x_test, y_test)
        print("acc: ", baseline_model_accuracy)
        print("fb score: ", fb)
        fbs.append(fb)
        cvscores.append(fb * 100)

        # save the best model for this fold
        fb *= 10000
        new_model.save("./trained_models/model_best_%s_%d_%d.h5" %(name_post, fb, i))
        print("model_best_%s_%d_%d.h5 saved!" %(name_post, fb, i))

        i += 1
    
    # print cross validation result
    print("fbs: ", fbs)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


if __name__ == '__main__':
    main()
