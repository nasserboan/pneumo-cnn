from keras import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.metrics import AUC, FalseNegatives, FalsePositives, TrueNegatives, TruePositives
from keras.callbacks import ModelCheckpoint, EarlyStopping




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
        
        model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape = self.input_shape))       
        model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(self.dropout))

        ## final layer
        if self.n_classes <= 2:
            model.add(Dense(self.n_classes,activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC(),FalseNegatives(),FalsePositives(),TruePositives(),TrueNegatives(),'accuracy'])
        else:
            model.add(Dense(self.n_classes,activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[AUC(),FalseNegatives(),'accuracy'])
        

        print(model.summary())

        return model

    def train(self):

        check = ModelCheckpoint(filepath='lenet/saved_model/weights.hdf5',save_best_only=True,mode='min',monitor='false_negatives_1')
        
        model = self._define_model()
        model.fit(self.x_train,self.y_train,
                batch_size=self.batch_size, epochs=self.epochs,
                verbose=self.verbose,
                validation_data=(self.x_test,self.y_test),
                callbacks=[check])

        with open('lenet/saved_model/model.json','w') as file:
            file.write(model.to_json())

        return model