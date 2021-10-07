from keras.preprocessing.image import ImageDataGenerator as Imgen
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Dropout,Conv2D,GlobalAveragePooling2D,Flatten
from tensorflow import keras
from tensorflow.python.keras.applications.xception import Xception
import pandas as pd
import os

from tensorflow.python.keras.layers import BatchNormalization

data_location = "C:\\Users\\18053\\Desktop\\Breed Detection\\data"

# for root, dir, files in os.walk(data_location):
#     for filename in files:
#         print(os.path.join(root,filename))


train_dir = data_location + "\\train"
test_dir = data_location + "\\test"

train_size = len(os.listdir(train_dir))
test_size = len(os.listdir(test_dir))

sample_submission = pd.read_csv(os.path.join(data_location,"sample_submission.csv"))
train_data = pd.read_csv(os.path.join(data_location,"labels.csv"))
# print(train_data.head())

train_data["id"] += ".jpg"
print(train_data["id"])

preprocesseddata = Imgen(preprocessing_function = keras.applications.nasnet.preprocess_input,
                      shear_range = 0.2,
                     horizontal_flip = True,
                     validation_split = 0.1)
train = preprocesseddata.flow_from_dataframe(train_data,
directory = train_dir,
x_col = 'id',
y_col = 'breed',
subset = 'training',
color_mode = 'rgb',
class_mode = 'categorical',
target_size = (256,256),
batch_size = 32,
shuffle =True,
seed = 123)

val_ds = preprocesseddata.flow_from_dataframe(
train_data,
directory = train_dir,
x_col = 'id',
y_col = 'breed',
subset = 'validation',
color_mode = 'rgb',
class_mode = 'categorical',
target_size = (256,256),
batch_size = 32,
shuffle =True,
seed = 123)

names = train.class_indices

class_names = [k for k,v in names.items()]

inceptionv3_model = InceptionV3(include_top = False,
                                   weights ='imagenet',
                                  input_shape = (256,256,3)
                                  )
inceptionv3_model.trainable = False

xception_model = Xception(include_top=False, pooling='avg', weights="imagenet")
xception_model.trainable = False

model = Sequential()
model.add(inceptionv3_model)
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.4))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(120,activation = 'softmax'))

print(model.summary())

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])



callback = [keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2),
           keras.callbacks.ModelCheckpoint("InceptionV3.h5",save_best_only =True,verbose =2)]

history = model.fit(train,epochs = 25,validation_data = val_ds,callbacks = callback)