''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
'''

from __future__ import print_function
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from keras import backend as K
from DataGenerator import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *

class TabCNN:
    
    def __init__(self, 
                 batch_size=128, 
                 epochs=8,
                 con_win_size = 9,
                 spec_repr="c",
                 data_path="../data/spec_repr/",
                 id_file="id.csv",
                 save_path="saved/"):   
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        
        self.load_IDs()
        
        self.save_folder = self.save_path + self.spec_repr + " " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = self.save_folder + "log.txt"
        
        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["data"] = ["g0","g1","g2","g3","g4","g5","mean","std dev"]
        
        if self.spec_repr == "c":
            self.input_shape = (192, self.con_win_size, 1)
        elif self.spec_repr == "m":
            self.input_shape = (128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.input_shape = (320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.input_shape = (1025, self.con_win_size, 1)
            
        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

    def load_IDs(self):
        csv_file = self.data_path + self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])
        
    def partition_data(self, data_split):
        self.data_split = data_split
        self.partition = {}
        self.partition["training"] = []
        self.partition["validation"] = []
        for ID in self.list_IDs:
            guitarist = int(ID.split("_")[0])
            if guitarist == data_split:
                self.partition["validation"].append(ID)
            else:
                self.partition["training"].append(ID)
                
        self.training_generator = DataGenerator(self.partition['training'], 
                                                data_path=self.data_path, 
                                                batch_size=self.batch_size, 
                                                shuffle=True,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size)
        
        self.validation_generator = DataGenerator(self.partition['validation'], 
                                                data_path=self.data_path, 
                                                batch_size=len(self.partition['validation']), 
                                                shuffle=False,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size)
        
        self.split_folder = self.save_folder + str(self.data_split) + "/"
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)
                
    def log_model(self):
        with open(self.log_file,'w') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
       
    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
        return K.concatenate(string_sm, axis=1)
    
    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
        return loss
    
    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
           
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))   
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings)) # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        model.compile(loss=self.catcross_by_string,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=[self.avg_acc])
        
        self.model = model

    def train(self):
        self.model.fit_generator(generator=self.training_generator,
                    validation_data=None,
                    epochs=self.epochs,
                    verbose=1,
                    use_multiprocessing=True,
                    workers=9)
        
    def save_weights(self):
        self.model.save_weights(self.split_folder + "weights.h5")
        
    def test(self):
        self.X_test, self.y_gt = self.validation_generator[0]
        self.y_pred = self.model.predict(self.X_test)
        
    def save_predictions(self):
        np.savez(self.split_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)
        
    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))
        
    def save_results_csv(self):
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        output["data"] =  self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder + "results.csv") 
        
##################################
########### EXPERIMENT ###########
##################################

tabcnn = TabCNN()

print("logging model...")
tabcnn.build_model()
tabcnn.log_model()

for fold in range(6):
    print("\nfold " + str(fold))
    tabcnn.partition_data(fold)
    print("building model...")
    tabcnn.build_model()
    print("training...")
    tabcnn.train()
    tabcnn.save_weights()
    print("testing...")
    tabcnn.test()
    tabcnn.save_predictions()
    print("evaluation...")
    tabcnn.evaluate()
print("saving results...")
tabcnn.save_results_csv()
