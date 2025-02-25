from scipy.io import loadmat 
import os
import glob
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold 
# from sklearn.preprocessing import StandardScaler 
from tensorflow import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt 
import seaborn as sns
from keras.regularizers import l2 

PreSleepfeaturesFolderPath = 'D:\\_temp_matlab_R2024b_Windows\\matlab\\LRHWakeClassifier\\imagery_presleep\\features' 
PosSleepfeaturesFolderPath = 'D:\\_temp_matlab_R2024b_Windows\\matlab\\LRHWakeClassifier\\imagery_postsleep\\features'

PreSleeplabelsFolderPath = 'D:\\_temp_matlab_R2024b_Windows\\matlab\\LRHWakeClassifier\\imagery_presleep\\labels' 
PosSleeplabelsFolderPath = 'D:\\_temp_matlab_R2024b_Windows\\matlab\\LRHWakeClassifier\\imagery_postsleep\\labels'

PreSleeplabelsallFiles = glob.glob(os.path.join(PreSleeplabelsFolderPath, '*.mat'))
PosSleeplabelsallfiles = glob.glob(os.path.join(PosSleeplabelsFolderPath, '*.mat'))

PreSleepfeaturesallFiles = glob.glob(os.path.join(PreSleepfeaturesFolderPath, '*.mat'))
PosSfeaturesallfiles = glob.glob(os.path.join(PosSleepfeaturesFolderPath, '*.mat'))

All_Prefeatures =[]
All_Prelabels =[]
ALL_Posfeatures =[]
All_Poslabels =[]
all_participant_acc =[]

for preFilePath, preLabelFilePath, posFilePath, posLabelFilePath in zip(PreSleepfeaturesallFiles, PreSleeplabelsallFiles, PosSfeaturesallfiles, PosSleeplabelsallfiles):    
    Prefeatures = loadmat(preFilePath)['features'] 
    Prefeatures = np.transpose(Prefeatures, (0, 2, 1))            
    # labelssss
    Prelabels = loadmat(preLabelFilePath)['labels']  
    Prelabels[Prelabels == 2] = 1 
    Prelabels[Prelabels == 4] = 0 
    Prelabels[Prelabels == 3] = 0   
    Prefeatures = Prefeatures.astype(np.float32)
    Prelabels = Prelabels.astype(np.float32)

    Pre_X_Train, Pre_X_Val, Pre_y_Train, Pre_y_Val = train_test_split(Prefeatures, Prelabels, test_size=0.1, random_state=1)   
    
   # lstm_input = Input(shape=(Pre_X_Train.shape[1], Pre_X_Train.shape[2]), name='lstm_input')
   # Inputs = LSTM(200, name='first_layer', return_sequences=True, kernel_regularizer=l2(0.08))(lstm_input)
    #Inputs = BatchNormalization()(Inputs)
    #Inputs = Dropout(0.3)(Inputs)
    #Inputs = LSTM(150, name='SEC_layer', return_sequences=True, kernel_regularizer=l2(0.08))(Inputs)
    #Inputs = BatchNormalization()(Inputs)
    #Inputs = Dropout(0.4)(Inputs)
    #Inputs = LSTM(100, name='THR_layer', return_sequences=True, kernel_regularizer=l2(0.08))(Inputs)
    #Inputs = BatchNormalization()(Inputs)
    #Inputs = Dropout(0.5)(Inputs)
    #Inputs = LSTM(50, name='FOURTH_layer', kernel_regularizer=l2(0.08))(Inputs)
    #Inputs = BatchNormalization()(Inputs)
    #Inputs = Dropout(0.6)(Inputs)
    #Inputs = Dense(1, activation='sigmoid', name='dense_layer')(Inputs)
    #model = Model(inputs=lstm_input, outputs=Inputs)

    # Compile the model
    #adam = optimizers.Adam(learning_rate=0.001)
    #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    # Train the model
    #model.fit(Pre_X_Train, Pre_y_Train, validation_data=(Pre_X_Val, Pre_y_Val), epochs=50, batch_size=32, callbacks=[checkpoint])
    #model.load_weights('best_model.keras')
    #############################
    #lstm_input = Input(shape=(Pre_X_Train.shape[1], Pre_X_Train.shape[2]), name='lstm_input') 
   # Inputs = LSTM(200, name='first_layer', return_sequences=True, kernel_regularizer=l2(0.03))(lstm_input)
    #Inputs = BatchNormalization()(Inputs)
    #Inputs = Dropout(0.3)(Inputs)
    #Inputs = LSTM(150, name='SEC_layer', return_sequences=True, kernel_regularizer=l2(0.03))(Inputs)   
    #Inputs = Dropout(0.3)(Inputs)
    #Inputs = LSTM(100, name='THR_layer', return_sequences=True, kernel_regularizer=l2(0.03))(Inputs)   
    #Inputs = Dropout(0.3)(Inputs)
    #Inputs = LSTM(50, name='FOURTH_layer', kernel_regularizer=l2(0.03))(Inputs)   
    #Inputs = BatchNormalization()(Inputs)
    #Inputs = Dropout(0.7)(Inputs)
    #Inputs = Dense(1, activation='sigmoid',name='dense_layer')(Inputs) 
    #model = Model(inputs=lstm_input, outputs=Inputs) 
    #adam = optimizers.Adam(learning_rate=0.001) 
    #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])     
   # model.fit(Pre_X_Train, Pre_y_Train, validation_data=(Pre_X_Val, Pre_y_Val), epochs=10, batch_size=32)      #, callbacks=[early_stopping]      


    lstm_input = Input(shape=(Pre_X_Train.shape[1], Pre_X_Train.shape[2]), name='lstm_input')
    Inputs = LSTM(20, name='first_layer', return_sequences=True)(lstm_input) #   
    #Inputs = LSTM(150, name='SEC_layer', return_sequences=True)(Inputs)    
    #Inputs = LSTM(100, name='THR_layer', return_sequences=True)(Inputs)    
    #
    Inputs = Dropout(0.2)(Inputs)  
    Inputs = LSTM(10, name='sec_layer', return_sequences=True)(Inputs) 
    #Inputs = Dropout(0.1)(Inputs) 
    Inputs = LSTM(5, name='third_layer')(Inputs)     
    Inputs = Dense(1, activation='sigmoid', name='dense_layer')(Inputs)
    model = Model(inputs=lstm_input, outputs=Inputs)    
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])    
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')   
    model.fit(Pre_X_Train, Pre_y_Train, validation_data=(Pre_X_Val, Pre_y_Val), epochs=15, batch_size=32, callbacks=[checkpoint])
    model.load_weights('best_model.keras')

    PosFeatures = loadmat(posFilePath)['features'] 
    PosFeatures = np.transpose(PosFeatures, (0, 2, 1)) 
    #labeels 
    Poslabels = loadmat(posLabelFilePath)['labels']
    Poslabels[Poslabels == 2] = 1 
    Poslabels[Poslabels == 4] = 0 
    Poslabels[Poslabels == 3] = 0   
    PosFeatures = PosFeatures.astype(np.float32)
    Poslabels = Poslabels.astype(np.float32) 
    loss, accuracy = model.evaluate(PosFeatures, Poslabels)
    all_participant_acc.append(accuracy)

all_participant_acc = np.array(all_participant_acc) 
mean_acc = np.mean(all_participant_acc) 
std_error_acc = np.std(all_participant_acc) / np.sqrt(len(all_participant_acc)) 
fig, ax = plt.subplots() 
ax.errorbar(x=[0], y=[mean_acc], yerr=[std_error_acc], fmt='o', color='r', capsize=5) 
sns.violinplot(data=all_participant_acc, ax=ax, inner=None, color=".9") 
ax.scatter(np.full(len(all_participant_acc), 0), all_participant_acc, color='blue', zorder=2)
ax.set_title('Violin Plot with Error Bar') 
ax.set_ylabel('Accuracy') 
plt.show(block=True)
plt.savefig('violin_plot_with_error_bar.png')