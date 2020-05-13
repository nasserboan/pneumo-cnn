from lenet.train_lenet  import MyLeNet
from alexnet.train_alex import MyAlexNet
from vgg.train_vgg import MyVGGNet

from dataload.loader import DataLoader


from silence_tensorflow import silence_tensorflow
silence_tensorflow()

# (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
# x_valid = x_valid.reshape(10000, 28, 28, 1).astype('float32')

# x_train /= 255
# x_valid /= 255

# n_classes = 10
# y_train = to_categorical(y_train, n_classes)
# y_valid = to_categorical(y_valid, n_classes)

# model = MyLeNet(input_shape=(28,28,1),n_classes=10,
#                 dropout=0.4,epochs=20,
#                 x_train = x_train,y_train = y_train,
#                 x_test = x_valid,y_test = y_valid,
#                 batch_size=32)

# model.train()

EPOCHS = 200
DATA_SHAPE_MODEL = (200,200,3)
DATA_SHAPE_LOADER = (200,200)
N_CLASSES = 2
BATCH_SIZE = 32
DROP_OUT = 0.1

loader = DataLoader(train_samples=512,test_samples=112,n_classes= N_CLASSES ,data_shape = DATA_SHAPE_LOADER)
x_train,y_train = loader.load_train()
x_test, y_test = loader.load_test()

## ==========================================================================

le = MyLeNet(input_shape = DATA_SHAPE_MODEL ,n_classes = N_CLASSES,
               dropout= DROP_OUT, epochs= EPOCHS,
               x_train= x_train, y_train = y_train,
               x_test = x_test, y_test = y_test,
               batch_size = BATCH_SIZE)

le.train()

del le

## ==========================================================================

alex = MyAlexNet(input_shape = DATA_SHAPE_MODEL ,n_classes = N_CLASSES,
               dropout= DROP_OUT, epochs= EPOCHS,
               x_train= x_train, y_train = y_train,
               x_test = x_test, y_test = y_test,
               batch_size = BATCH_SIZE)

alex.train()

del alex

## ==========================================================================

vgg = MyVGGNet(input_shape = DATA_SHAPE_MODEL ,n_classes = N_CLASSES,
               dropout= DROP_OUT, epochs= EPOCHS,
               x_train= x_train, y_train = y_train,
               x_test = x_test, y_test = y_test,
               batch_size = BATCH_SIZE)

vgg.train()

del vgg