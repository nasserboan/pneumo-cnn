from dataload import loader
from vgg import train_vgg
import matplotlib.pyplot as plt
import cv2

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

EPOCHS = 100
DATA_SHAPE_MODEL = (224,224,3)
DATA_SHAPE_LOADER = (224,224)
N_CLASSES = 2
BATCH_SIZE = 32
DROP_OUT = 0.6

loader = loader.DataLoader(train_samples=1000,test_samples=200,n_classes= N_CLASSES ,data_shape = DATA_SHAPE_LOADER)

x_train,y_train = loader.load_train()
x_test, y_test = loader.load_test()

x_train = x_train / 255
x_test = x_test / 255

vgg = train_vgg.MyVGGNet(input_shape = DATA_SHAPE_MODEL ,n_classes = N_CLASSES,
               dropout= DROP_OUT, epochs= EPOCHS,
               x_train= x_train, y_train = y_train,
               x_test = x_test, y_test = y_test,
               batch_size = BATCH_SIZE,verbose=1)

curves = vgg.train()
data = curves.history.history

f, ax = plt.subplots(1,3,figsize=(20,9))
plt.suptitle('Avaliação da VGG',y=0.93,size=25)

ax[0].plot(list(range(1,EPOCHS+1)),data['loss'],color='tab:blue',label = 'train loss')
ax[0].plot(list(range(1,EPOCHS+1)),data['val_loss'],color='tab:orange',label = 'test loss')
ax[0].grid(which='major',axis='x',alpha=0.2)
ax[0].set_xticks(list(range(1,EPOCHS+1)))
ax[0].set_xticklabels(labels = list(range(1,EPOCHS+1)),rotation=45)
ax[0].legend()

# ax[1].plot(list(range(1,EPOCHS+1)),data['auc_1'],color='tab:blue',label = 'train AUC')
# ax[1].plot(list(range(1,EPOCHS+1)),data['val_auc_1'],color='tab:orange',label = 'test AUC')
# ax[1].grid(which='major',axis='x',alpha=0.2)
# ax[1].set_xticks(list(range(1,EPOCHS+1)))
# ax[1].set_xticklabels(labels = list(range(1,EPOCHS+1)),rotation=45)
# ax[1].legend()

# ax[2].plot(list(range(1,EPOCHS+1)),data['false_negatives_1'],color='tab:blue',label = 'train false negatives')
# ax[2].plot(list(range(1,EPOCHS+1)),data['val_false_negatives_1'],color='tab:orange',label = 'test false negatives')
# ax[2].grid(which='major',axis='x',alpha=0.2)
# ax[2].set_xticks(list(range(1,EPOCHS+1)))
# ax[2].set_xticklabels(labels = list(range(1,EPOCHS+1)),rotation=45)
# ax[2].legend()

plt.savefig('vgg/vgg_train.png')