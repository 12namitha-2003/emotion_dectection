{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "909cd1d0-7eb2-4664-8f88-5243bec1a526",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 12318 images belonging to 4 classes.\n"
     ]
    },
    {
     "ename": "FileNotFoundError",
     "evalue": "[WinError 3] The system cannot find the path specified: 'data_cleaned/test'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mFileNotFoundError\u001b[39m                         Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 21\u001b[39m\n\u001b[32m     11\u001b[39m test_datagen = ImageDataGenerator(rescale=\u001b[32m1.\u001b[39m/\u001b[32m255\u001b[39m)\n\u001b[32m     13\u001b[39m train_set = train_datagen.flow_from_directory(\n\u001b[32m     14\u001b[39m     train_path,\n\u001b[32m     15\u001b[39m     target_size=(\u001b[32m48\u001b[39m, \u001b[32m48\u001b[39m),\n\u001b[32m   (...)\u001b[39m\u001b[32m     18\u001b[39m     class_mode=\u001b[33m\"\u001b[39m\u001b[33mcategorical\u001b[39m\u001b[33m\"\u001b[39m\n\u001b[32m     19\u001b[39m )\n\u001b[32m---> \u001b[39m\u001b[32m21\u001b[39m test_set = \u001b[43mtest_datagen\u001b[49m\u001b[43m.\u001b[49m\u001b[43mflow_from_directory\u001b[49m\u001b[43m(\u001b[49m\n\u001b[32m     22\u001b[39m \u001b[43m    \u001b[49m\u001b[43mtest_path\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m     23\u001b[39m \u001b[43m    \u001b[49m\u001b[43mtarget_size\u001b[49m\u001b[43m=\u001b[49m\u001b[43m(\u001b[49m\u001b[32;43m48\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[32;43m48\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m     24\u001b[39m \u001b[43m    \u001b[49m\u001b[43mcolor_mode\u001b[49m\u001b[43m=\u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mgrayscale\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[32m     25\u001b[39m \u001b[43m    \u001b[49m\u001b[43mbatch_size\u001b[49m\u001b[43m=\u001b[49m\u001b[32;43m32\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[32m     26\u001b[39m \u001b[43m    \u001b[49m\u001b[43mclass_mode\u001b[49m\u001b[43m=\u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mcategorical\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\n\u001b[32m     27\u001b[39m \u001b[43m)\u001b[49m\n\u001b[32m     29\u001b[39m \u001b[38;5;66;03m# Build CNN model\u001b[39;00m\n\u001b[32m     30\u001b[39m model = Sequential([\n\u001b[32m     31\u001b[39m     Conv2D(\u001b[32m32\u001b[39m, (\u001b[32m3\u001b[39m,\u001b[32m3\u001b[39m), activation=\u001b[33m'\u001b[39m\u001b[33mrelu\u001b[39m\u001b[33m'\u001b[39m, input_shape=(\u001b[32m48\u001b[39m,\u001b[32m48\u001b[39m,\u001b[32m1\u001b[39m)),\n\u001b[32m     32\u001b[39m     MaxPooling2D(\u001b[32m2\u001b[39m,\u001b[32m2\u001b[39m),\n\u001b[32m   (...)\u001b[39m\u001b[32m     40\u001b[39m     Dense(train_set.num_classes, activation=\u001b[33m'\u001b[39m\u001b[33msoftmax\u001b[39m\u001b[33m'\u001b[39m)\n\u001b[32m     41\u001b[39m ])\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~\\miniconda3\\Lib\\site-packages\\keras\\src\\legacy\\preprocessing\\image.py:1136\u001b[39m, in \u001b[36mImageDataGenerator.flow_from_directory\u001b[39m\u001b[34m(self, directory, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation, keep_aspect_ratio)\u001b[39m\n\u001b[32m   1118\u001b[39m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34mflow_from_directory\u001b[39m(\n\u001b[32m   1119\u001b[39m     \u001b[38;5;28mself\u001b[39m,\n\u001b[32m   1120\u001b[39m     directory,\n\u001b[32m   (...)\u001b[39m\u001b[32m   1134\u001b[39m     keep_aspect_ratio=\u001b[38;5;28;01mFalse\u001b[39;00m,\n\u001b[32m   1135\u001b[39m ):\n\u001b[32m-> \u001b[39m\u001b[32m1136\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mDirectoryIterator\u001b[49m\u001b[43m(\u001b[49m\n\u001b[32m   1137\u001b[39m \u001b[43m        \u001b[49m\u001b[43mdirectory\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1138\u001b[39m \u001b[43m        \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[32m   1139\u001b[39m \u001b[43m        \u001b[49m\u001b[43mtarget_size\u001b[49m\u001b[43m=\u001b[49m\u001b[43mtarget_size\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1140\u001b[39m \u001b[43m        \u001b[49m\u001b[43mcolor_mode\u001b[49m\u001b[43m=\u001b[49m\u001b[43mcolor_mode\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1141\u001b[39m \u001b[43m        \u001b[49m\u001b[43mkeep_aspect_ratio\u001b[49m\u001b[43m=\u001b[49m\u001b[43mkeep_aspect_ratio\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1142\u001b[39m \u001b[43m        \u001b[49m\u001b[43mclasses\u001b[49m\u001b[43m=\u001b[49m\u001b[43mclasses\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1143\u001b[39m \u001b[43m        \u001b[49m\u001b[43mclass_mode\u001b[49m\u001b[43m=\u001b[49m\u001b[43mclass_mode\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1144\u001b[39m \u001b[43m        \u001b[49m\u001b[43mdata_format\u001b[49m\u001b[43m=\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[43mdata_format\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1145\u001b[39m \u001b[43m        \u001b[49m\u001b[43mbatch_size\u001b[49m\u001b[43m=\u001b[49m\u001b[43mbatch_size\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1146\u001b[39m \u001b[43m        \u001b[49m\u001b[43mshuffle\u001b[49m\u001b[43m=\u001b[49m\u001b[43mshuffle\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1147\u001b[39m \u001b[43m        \u001b[49m\u001b[43mseed\u001b[49m\u001b[43m=\u001b[49m\u001b[43mseed\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1148\u001b[39m \u001b[43m        \u001b[49m\u001b[43msave_to_dir\u001b[49m\u001b[43m=\u001b[49m\u001b[43msave_to_dir\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1149\u001b[39m \u001b[43m        \u001b[49m\u001b[43msave_prefix\u001b[49m\u001b[43m=\u001b[49m\u001b[43msave_prefix\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1150\u001b[39m \u001b[43m        \u001b[49m\u001b[43msave_format\u001b[49m\u001b[43m=\u001b[49m\u001b[43msave_format\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1151\u001b[39m \u001b[43m        \u001b[49m\u001b[43mfollow_links\u001b[49m\u001b[43m=\u001b[49m\u001b[43mfollow_links\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1152\u001b[39m \u001b[43m        \u001b[49m\u001b[43msubset\u001b[49m\u001b[43m=\u001b[49m\u001b[43msubset\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1153\u001b[39m \u001b[43m        \u001b[49m\u001b[43minterpolation\u001b[49m\u001b[43m=\u001b[49m\u001b[43minterpolation\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1154\u001b[39m \u001b[43m        \u001b[49m\u001b[43mdtype\u001b[49m\u001b[43m=\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[43mdtype\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m   1155\u001b[39m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~\\miniconda3\\Lib\\site-packages\\keras\\src\\legacy\\preprocessing\\image.py:456\u001b[39m, in \u001b[36mDirectoryIterator.__init__\u001b[39m\u001b[34m(self, directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation, keep_aspect_ratio, dtype)\u001b[39m\n\u001b[32m    454\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m classes:\n\u001b[32m    455\u001b[39m     classes = []\n\u001b[32m--> \u001b[39m\u001b[32m456\u001b[39m     \u001b[38;5;28;01mfor\u001b[39;00m subdir \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28msorted\u001b[39m(\u001b[43mos\u001b[49m\u001b[43m.\u001b[49m\u001b[43mlistdir\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdirectory\u001b[49m\u001b[43m)\u001b[49m):\n\u001b[32m    457\u001b[39m         \u001b[38;5;28;01mif\u001b[39;00m os.path.isdir(os.path.join(directory, subdir)):\n\u001b[32m    458\u001b[39m             classes.append(subdir)\n",
      "\u001b[31mFileNotFoundError\u001b[39m: [WinError 3] The system cannot find the path specified: 'data_cleaned/test'"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout\n",
    "\n",
    "# Paths to cleaned data\n",
    "train_path = \"data_cleaned/train\"\n",
    "test_path = \"data_cleaned/test\"\n",
    "\n",
    "# Image preprocessing\n",
    "train_datagen = ImageDataGenerator(rescale=1./255)\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "train_set = train_datagen.flow_from_directory(\n",
    "    train_path,\n",
    "    target_size=(48, 48),\n",
    "    color_mode=\"grayscale\",\n",
    "    batch_size=32,\n",
    "    class_mode=\"categorical\"\n",
    ")\n",
    "\n",
    "test_set = test_datagen.flow_from_directory(\n",
    "    test_path,\n",
    "    target_size=(48, 48),\n",
    "    color_mode=\"grayscale\",\n",
    "    batch_size=32,\n",
    "    class_mode=\"categorical\"\n",
    ")\n",
    "\n",
    "# Build CNN model\n",
    "model = Sequential([\n",
    "    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),\n",
    "    MaxPooling2D(2,2),\n",
    "    Conv2D(64, (3,3), activation='relu'),\n",
    "    MaxPooling2D(2,2),\n",
    "    Conv2D(128, (3,3), activation='relu'),\n",
    "    MaxPooling2D(2,2),\n",
    "    Flatten(),\n",
    "    Dense(128, activation='relu'),\n",
    "    Dropout(0.5),\n",
    "    Dense(train_set.num_classes, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "model.summary()\n",
    "\n",
    "# Train model\n",
    "history = model.fit(train_set, validation_data=test_set, epochs=15)\n",
    "\n",
    "# Save model\n",
    "model.save(\"emotion_detection_model.h5\")\n",
    "print(\"\\n✅ Model training complete and saved as 'emotion_detection_model.h5'\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a4de7f8-f53f-4dca-8349-976b5e1fc3ff",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
