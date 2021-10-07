from keras.preprocessing.image import ImageDataGenerator as Imgen
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Dropout,Conv2D,GlobalAveragePooling2D,Flatten
from tensorflow import keras
from tensorflow.python.keras import Input
import pandas as pd
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.layers import BatchNormalization, Lambda
from tensorflow.python.keras.models import Model
import numpy as np

data_location = "C:\\Users\\18053\\Desktop\\Breed Detection\\data"

# for root, dir, files in os.walk(data_location):
#     for filename in files:
#         print(os.path.join(root,filename))

def get_features(model_name, model_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    # Extract feature.
    feature_maps = feature_extractor.predict(data, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps

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

inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3,
                                  inception_preprocessor,
                                  (256, 256, 3), train)
xception_preprocessor = preprocess_input
xception_features = get_features(Xception,
                                 xception_preprocessor,
                                 (256, 256, 3), train)
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2,
                                   inc_resnet_preprocessor,
                                   (256, 256, 3), train)


final_features = np.concatenate([inception_features,
                                 xception_features,
                                 inc_resnet_features,], axis=-1)

print(final_features.shape)

model = Sequential()
model.add(Dropout(0.4, input_shape=(final_features.shape[1])))
model.add(Dense(120, activation="softmax"))


print(model.summary())

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])



callback = [keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2),
           keras.callbacks.ModelCheckpoint("Xception.h5",save_best_only =True,verbose =2)]

history = model.fit(train,epochs = 25,validation_data = val_ds,callbacks = callback)