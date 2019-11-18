from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import glob
import shutil
from keras.preprocessing.image import ImageDataGenerator
model = Sequential()
train_filename = 'flickr_logos_27_dataset_training_set_annotation.txt'
train_img_name = 'flickr_logos_27_dataset_images'
train_imgs = os.listdir(train_img_name)
PATH = 'train'
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 32
def img_processing(filename, img_folder, logo):
    i = 0
    os.makedirs(PATH)
    file = open(filename, 'r').read().split('\n')
    print(file)
    img_dict = {}
    for i in range(len(file)):
        img_dict.update({file[i].split(' ')[0]: file[i].split(' ')[1:-1]})
    print(img_dict)
    for img in glob.glob(img_folder + '/*.jpg'):
        print("Parsing %s" % img)
        check = str(img)[str(img).index('\\') + 1:]
        if check in img_dict and img_dict[check][0] == logo:
            shutil.copy(img, PATH)


def extract_imgstotrain(filename, img_folder):
    file = open(filename, 'r').read().split('\n')
    print(file)
    img_dict = {}
    for i in range(len(file)):
        img_dict.update({file[i].split(' ')[0]: file[i].split(' ')[1:-1]})
    print(img_dict)
    for img in glob.glob(img_folder+'/*.jpg'):
        print("Parsing %s" % img)
        check = str(img)[str(img).index('\\')+1:]
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=90, horizontal_flip=True, vertical_flip=True)
        train_generator = train_datagen.flow_from_directory(PATH, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)
        imag = load_img(img, target_size=(224, 224))
def build_model():
    base = ResNet50(weights='imagenet',include_top=False)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    preds = Dense(27,activation='softmax')(x)
    return Model(inputs=base.input, outputs=preds)

def train():
    model = build_model()

    for layer in model.layers[:176]:
        layer.trainable=False
    for layer in model.layers[176:]:
        layer.trainable=True

    for i,layer in enumerate(model.layers):
      print(i,layer.name, layer.trainable)
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=90, horizontal_flip=True, vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(PATH, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit_generator(generator=train_generator, steps_per_epoch=10, epochs=40)
    model.save_weights('aug_model.h5')

parse_predictions = []
for dir in glob.glob(PATH+'/*'):
    print("Parsing %s" % dir)
    parse_predictions.append(str(dir)[6:])
parse_predictions.sort()
print(parse_predictions)

def predict():
    model = build_model()
    model.load_weights('aug_model.h5')
    image = load_img('apple.png', target_size=(224, 224))
    image.show()
    image = img_to_array(image)
    # reshape data for the model
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the ResNet50 model
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # predict the probability across all output classes
    yhat = model.predict(image)
    print(yhat)
    max_val = np.argmax(yhat)
    if yhat[0][max_val] * 100 >= 5:
        print('%s (%.2f%%)' % (parse_predictions[max_val], yhat[0][max_val] * 100))
    else:
        print("no logo detected")
predict()