import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


##### DOWNLOADING AND PREPARING DATASET ######

train_gen = ImageDataGenerator(rescale=1/255)
train_imgs= tf.keras.utils.image_dataset_from_directory('/Users/nadia/PycharmProjects/pythonFinalProject/Vegetable Images/train')
train_set = train_gen.flow_from_directory('/Users/nadia/PycharmProjects/pythonFinalProject/Vegetable Images/train', target_size=(224,224), batch_size=32, class_mode='categorical')

validation_gen = ImageDataGenerator(rescale=1/255)
validation_imgs= tf.keras.preprocessing.image_dataset_from_directory('/Users/nadia/PycharmProjects/pythonFinalProject/Vegetable Images/validation')
validation_set = validation_gen.flow_from_directory('/Users/nadia/PycharmProjects/pythonFinalProject/Vegetable Images/validation', target_size=(224,224), batch_size=32, class_mode='categorical')

test_gen = ImageDataGenerator(rescale=1/255)
test_imgs = tf.keras.utils.image_dataset_from_directory('/Users/nadia/PycharmProjects/pythonFinalProject/Vegetable Images/test')
test_set = test_gen.flow_from_directory('/Users/nadia/PycharmProjects/pythonFinalProject/Vegetable Images/test', target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)


##### VERIFYING THE DATA ######

# creating labels to display data
class_names_tr = train_imgs.class_names
print(class_names_tr)

# displaying 25 images to verify data
plt.figure(figsize=(15, 15))
for img, lab in train_imgs.take(1):
    for i in range(25): # 25 images displayed
        ax = plt.subplot(5, 5, i + 1) # 5 rows, 5 columns
        plt.imshow(img[i].numpy().astype("uint8")) # [0..255] valid range for rgb integers
        plt.title(class_names_tr[lab[i]])
        plt.axis("off")
    plt.show()


##### CREATING THE CONVOLUTIONAL BASE ######

model = Sequential()

model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3), strides=1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3,3), activation='relu', strides=1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', strides=1))
model.add(layers.MaxPooling2D((2, 2)))


##### ADDING DENSE LAYERS ######

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(15, activation='softmax'))

model.summary()


##### COMPILING AND TRAINING MODEL ######

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics='accuracy')
history = model.fit(train_set, epochs=10, validation_data=validation_set)


##### EVALUATING MODEL ######

# plot accuracy curve
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

## plot loss curve
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()

# print loss and accuracy epochs summary
history_df = pd.DataFrame(history.history)
print(history_df)


####### CONFUSION MATRIX ##########

pred = tf.argmax(model.predict(test_set), axis=1)
classes = list(test_set.class_indices.keys())

pred_classes = [classes[x] for x in pred]
labels_classes = [classes[x] for x in test_set.labels]

# print precision and accuracy report
print(classification_report(labels_classes, pred_classes))

# display confusion matrix figure using seaborn sns
cf_matrix = confusion_matrix(labels_classes, pred_classes)
sns.set_theme(rc={'figure.figsize':(15,15)})
ax = sns.heatmap(cf_matrix, annot=True, cmap='crest', fmt="g", xticklabels=classes, yticklabels=classes, cbar=False)
ax.set_ylabel('True Labels')
ax.set_xlabel('Predicted Labels')
plt.show()

