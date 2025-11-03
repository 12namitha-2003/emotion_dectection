{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2e4206e8-d3da-4283-81ae-1fa84b2a78e8",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "❌ Test folder not found at: data_cleaned/test",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mFileNotFoundError\u001b[39m                         Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[2]\u001b[39m\u001b[32m, line 16\u001b[39m\n\u001b[32m     14\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mFileNotFoundError\u001b[39;00m(\u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[33m❌ Train folder not found at: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mtrain_path\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m\"\u001b[39m)\n\u001b[32m     15\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m os.path.exists(test_path):\n\u001b[32m---> \u001b[39m\u001b[32m16\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mFileNotFoundError\u001b[39;00m(\u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[33m❌ Test folder not found at: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mtest_path\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m\"\u001b[39m)\n\u001b[32m     18\u001b[39m \u001b[38;5;66;03m# -------------------------------\u001b[39;00m\n\u001b[32m     19\u001b[39m \u001b[38;5;66;03m# Image Preprocessing\u001b[39;00m\n\u001b[32m     20\u001b[39m \u001b[38;5;66;03m# -------------------------------\u001b[39;00m\n\u001b[32m     21\u001b[39m train_datagen = ImageDataGenerator(rescale=\u001b[32m1.0\u001b[39m / \u001b[32m255\u001b[39m)\n",
      "\u001b[31mFileNotFoundError\u001b[39m: ❌ Test folder not found at: data_cleaned/test"
     ]
    }
   ],
   "source": [
    "import os\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout\n",
    "\n",
    "# -------------------------------\n",
    "# Paths\n",
    "# -------------------------------\n",
    "train_path = \"data_cleaned/train\"\n",
    "test_path = \"data_cleaned/test\"\n",
    "\n",
    "# Check folder existence\n",
    "if not os.path.exists(train_path):\n",
    "    raise FileNotFoundError(f\"❌ Train folder not found at: {train_path}\")\n",
    "if not os.path.exists(test_path):\n",
    "    raise FileNotFoundError(f\"❌ Test folder not found at: {test_path}\")\n",
    "\n",
    "# -------------------------------\n",
    "# Image Preprocessing\n",
    "# -------------------------------\n",
    "train_datagen = ImageDataGenerator(rescale=1.0 / 255)\n",
    "test_datagen = ImageDataGenerator(rescale=1.0 / 255)\n",
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
    "# -------------------------------\n",
    "# Build CNN model\n",
    "# -------------------------------\n",
    "model = Sequential([\n",
    "    Conv2D(32, (3, 3), activation=\"relu\", input_shape=(48, 48, 1)),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(64, (3, 3), activation=\"relu\"),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(128, (3, 3), activation=\"relu\"),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Flatten(),\n",
    "    Dense(128, activation=\"relu\"),\n",
    "    Dropout(0.5),\n",
    "    Dense(train_set.num_classes, activation=\"softmax\")\n",
    "])\n",
    "\n",
    "model.compile(optimizer=\"adam\", loss=\"categorical_crossentropy\", metrics=[\"accuracy\"])\n",
    "model.summary()\n",
    "\n",
    "# -------------------------------\n",
    "# Train Model\n",
    "# -------------------------------\n",
    "print(\"\\n🚀 Starting model training...\")\n",
    "history = model.fit(train_set, validation_data=test_set, epochs=15)\n",
    "\n",
    "# -------------------------------\n",
    "# Save Model\n",
    "# -------------------------------\n",
    "model.save(\"emotion_detection_model.h5\")\n",
    "print(\"\\n✅ Model saved as 'emotion_detection_model.h5'\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "445a4a8e-740d-4342-bfea-d1e5e33624af",
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
