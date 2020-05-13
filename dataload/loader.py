import glob
import numpy as np
from keras.utils import to_categorical
from cv2 import imread
from cv2 import resize
from tqdm import tqdm


class DataLoader():
    def __init__(self,train_samples,test_samples,n_classes,data_shape):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.n_classes = n_classes
        self.data_shape = data_shape

    def load_train(self):
        print(f'Creating x_train with {self.train_samples*2} samples with {self.data_shape} shape.\n')

        x_train = np.array([resize(imread(file),self.data_shape) for file in tqdm(glob.glob("0data/train/PNEUMONIA/*.jpeg")[:self.train_samples])] + 
                           [resize(imread(file),self.data_shape) for file in tqdm(glob.glob("0data/train/NORMAL/*.jpeg")[:self.train_samples])])    

        print(f'x_train DONE! final shape : {x_train.shape}\n')

        print(f'Creating y_train with {self.train_samples*2} samples with {self.data_shape} shape.\n')
        y_train = np.array([1 for _ in tqdm(glob.glob("0data/train/PNEUMONIA/*.jpeg")[:self.train_samples])] + 
                           [0 for _ in tqdm(glob.glob("0data/train/NORMAL/*.jpeg")[:self.train_samples])])
        y_train = to_categorical(y_train, 2)

        print(f'y_train DONE! final shape : {y_train.shape}\n')

        return x_train, y_train

    def load_test(self):
        print(f'Creating x_test with {self.test_samples*2} samples with {self.data_shape} shape.\n')

        x_test = np.array([resize(imread(file),self.data_shape) for file in tqdm(glob.glob("0data/test/PNEUMONIA/*.jpeg")[:self.test_samples])] + 
                          [resize(imread(file),self.data_shape) for file in tqdm(glob.glob("0data/test/NORMAL/*.jpeg")[:self.test_samples])])

        print(f'x_test DONE! final shape : {x_test.shape}\n')

        print(f'Creating y_test with {self.test_samples*2} samples with {self.data_shape} shape.\n')
        y_test = np.array([1 for _ in tqdm(glob.glob("0data/test/PNEUMONIA/*.jpeg")[:self.test_samples])] + 
                          [0 for _ in tqdm(glob.glob("0data/test/NORMAL/*.jpeg")[:self.test_samples])])
        y_test = to_categorical(y_test, 2)
    
        print(f'y_train DONE! final shape : {y_test.shape}\n')

        return x_test, y_test
    