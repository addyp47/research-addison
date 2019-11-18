from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras
from keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import glob
import shutil
from keras.preprocessing.image import ImageDataGenerator
train_filename = 'flickr_logos_27_dataset_training_set_annotation.txt'
train_img_name = 'flickr_logos_27_dataset_images'
train_imgs = os.listdir(train_img_name)
PATH = 'train_with_boxes'
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 32
NB_CLASSES = 27
FREEZE_CUT_OFF = 249 #175 for RESNET50
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
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=90)
        train_generator = train_datagen.flow_from_directory(PATH, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)
        imag = load_img(img, target_size=(224, 224))
def train():
    base = ResNet50(input_shape=(HEIGHT, WIDTH, 3), weights='imagenet', include_top=False)
    x = Flatten(name='yolo_clf_0')(x)
    x = Dense(2048, activation='relu', name='yolo_clf_1')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5, name='yolo_clf_2')(x)

    # output tensor :
    # SS: Grid cells: 11*11
    # B: Bounding box per grid cell: 2
    # C: classes: 3
    # Coords: x, y, w, h per box: 4
    # tensor length: SS * (C +B(5) ) : 363--242--968 => 1573
    x = Dense(11 * 11 * (27 + 2 * 5), activation='linear', name='yolo_clf_3')(x)
    model = Model(inputs=base.input, outputs=preds)
    for layer in base.layers:
        layer.trainable = False
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=90, horizontal_flip=True, vertical_flip=True,
                                       validation_split=0.2)
    # train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, gvalidation_split=0.2)
    train_generator = train_datagen.flow_from_directory(PATH, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE,
                                                        subset='training', shuffle=True)
    validation_generator = train_datagen.flow_from_directory(PATH, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE,
                                                            subset='validation', shuffle=True)

    # model.load_weights('aug_model.h5')
    train_steps = train_generator.samples // BATCH_SIZE
    test_steps = validation_generator.samples // BATCH_SIZE
    for i,layer in enumerate(model.layers):
      print(i,layer.name, layer.trainable)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator, steps_per_epoch=10, validation_data=validation_generator, validation_steps=10, epochs=500)
    model.save('new_model_boxes.h5')

parse_predictions = []
for dir in glob.glob(PATH+'/*'):
    print("Parsing %s" % dir)
    parse_predictions.append(str(dir)[6:])
parse_predictions.sort()
print(parse_predictions)
def crop_logo(filepath, filename, img_folder, logo):
    i = 0
    logos = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'Heineken', 'HP', 'Intel', 'McDonalds', 'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']
    os.makedirs(filepath)
    file = open(filename, 'r').read().split('\n')
    print(file)
    img_dict = {}
    for i in range(len(file)):
        img_dict.update({file[i].split(' ')[0]: file[i].split(' ')[1:-1]})
    print(img_dict)
    num = 0
    for img in glob.glob(img_folder + '/*.jpg'):
        num += 1
        print("Parsing %s" % img)
        check = str(img)[str(img).index('\\') + 1:]
        if check in img_dict and img_dict[check][0] == logo:
            crop = []
            for val in img_dict[check][2:]:
                crop.append(int(val))
            if crop[0] == crop[2] or crop[1] == crop[3]:
                img2 = Image.open(img)
            else:
                img2 = Image.open(img).crop(tuple(crop))
            print(str(tuple(crop))+' '+str(img2.size))
            img2.save(os.path.join(filepath, str(logo) + '_' + str(num)) + '.JPG', 'JPEG', quality=90)

def predict():

    model = load_model('new_model_boxes.h5')
    image = load_img('apple2.png', target_size=(224, 224))
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
    if yhat[0][max_val]*100 >= 5:
        print('%s (%.2f%%)' % (parse_predictions[max_val], yhat[0][max_val]*100))
    else:
        print("no logo detected")
    #print(str(parse_predictions[max_val])+" "+str(yhat[0][max_val]*100))
    # convert the probabilities to class labels
    #label = parse_predictions[yhat]
    # retrieve the most likely result, e.g. highest probability
    #label = label[0][0]
    # print the classification
    #print('%s (%.2f%%)' % (label[1], label[2] * 100))'''
    # reshape data for the model
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the ResNet50 model
predict()
