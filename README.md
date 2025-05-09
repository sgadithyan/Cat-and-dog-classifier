# Cat-and-dog-classifier
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install kaggle\n",
    "!mkdir -p ~/.kaggle\n",
    "!kaggle datasets download -d aleemaparakatta/cats-and-dogs-mini-dataset\n",
    "!unzip cats-and-dogs-mini-dataset.zip -d data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import shutil\n",
    "\n",
    "unzipped_folder = [os.path.join('data', d) for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))][0]\n",
    "os.makedirs('data/train/cat', exist_ok=True)\n",
    "os.makedirs('data/train/dog', exist_ok=True)\n",
    "\n",
    "cats_src = os.path.join(unzipped_folder, 'cats_set')\n",
    "dogs_src = os.path.join(unzipped_folder, 'dogs_set')\n",
    "for fname in os.listdir(cats_src):\n",
    "    shutil.copy(os.path.join(cats_src, fname), 'data/train/cat')\n",
    "for fname in os.listdir(dogs_src):\n",
    "    shutil.copy(os.path.join(dogs_src, fname), 'data/train/dog')\n",
    "\n",
    "test_src = os.path.join(unzipped_folder, 'test_set')\n",
    "if os.path.isdir(test_src):\n",
    "    os.makedirs('data/test/test', exist_ok=True)\n",
    "    for fname in os.listdir(test_src):\n",
    "        shutil.copy(os.path.join(test_src, fname), 'data/test/test')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Building"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,\n",
    "                                   height_shift_range=0.2, horizontal_flip=True, validation_split=0.2)\n",
    "train_generator = train_datagen.flow_from_directory('data/train', target_size=(224,224), batch_size=32, class_mode='binary', subset='training', seed=42)\n",
    "validation_generator = train_datagen.flow_from_directory('data/train', target_size=(224,224), batch_size=32, class_mode='binary', subset='validation', seed=42)\n",
    "\n",
    "base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')\n",
    "base_model.trainable = False\n",
    "inputs = tf.keras.Input(shape=(224,224,3))\n",
    "x = base_model(inputs, training=False)\n",
    "x = tf.keras.layers.GlobalAveragePooling2D()(x)\n",
    "outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)\n",
    "model = tf.keras.Model(inputs, outputs)\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(train_generator, validation_data=validation_generator, epochs=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Prediction and Submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "test_generator = test_datagen.flow_from_directory('data/test', target_size=(224,224), batch_size=32, class_mode=None, shuffle=False)\n",
    "predictions = model.predict(test_generator)\n",
    "\n",
    "pred_labels = ['dog' if p>0.5 else 'cat' for p in predictions.ravel()]\n",
    "filenames = [os.path.basename(x) for x in test_generator.filenames]\n",
    "\n",
    "import pandas as pd\n",
    "submission = pd.DataFrame({'filename': filenames, 'prediction': pred_labels})\n",
    "submission.to_csv('submission.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
