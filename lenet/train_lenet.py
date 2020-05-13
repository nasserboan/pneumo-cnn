# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Flatten, Conv2D, MaxPooling2D


from keras import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.metrics import FalseNegatives, TrueNegatives, TruePositives, FalsePositives, AUC


import subprocess

class MyLeNet():
    def __init__(self,input_shape,n_classes,dropout,epochs,x_train,y_train,x_test,y_test,verbose=1,batch_size=128):
        ## data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        ## hyperparameters
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.dropout = dropout
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size

    def _define_model(self):
        
        model = Sequential()
        
        ## first convolution
        model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape = self.input_shape))
        
        ## second convolution
        model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
        
        ## maxpooling for reduction of activation maps
        model.add(MaxPooling2D(pool_size=(2,2)))

        ## dropout for preventing overfitting
        model.add(Dropout(self.dropout))

        ## flattening the data so can input it in a dense layer
        model.add(Flatten())

        ## dense layer for classification
        model.add(Dense(128,activation='relu'))

        ## more dropout for preventing overfitting
        model.add(Dropout(self.dropout*2))

        ## final layer
        if self.n_classes <= 2:
            model.add(Dense(self.n_classes,activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
        else:
            model.add(Dense(self.n_classes,activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[AUC()])
        

        print(model.summary())

        return model

    def train(self):

        # early_call = EarlyStopping(monitor='val_accuracy',
        #                            patience=round(self.epochs),
        #                            restore_best_weights=True,
        #                            verbose=1)

        # reduce_call = ReduceLROnPlateau(monitor='val_loss',
        #                            factor = 0.1,
        #                            patience=round(self.epochs/10),
        #                            min_lr=0.0001,
        #                            verbose=1)

        ## instantiation and training the model
        model = self._define_model()
        model.fit(self.x_train,self.y_train,
                batch_size=self.batch_size, epochs=self.epochs,
                verbose=self.verbose,
                validation_data=(self.x_test,self.y_test))#,
                #callbacks=[early_call])

# from keras import backend as k
# k.tensorflow_backend._get_available_gpus()

# from tensorflow_core._api.v2.config import get_visible_devices

# print(get_visible_devices())