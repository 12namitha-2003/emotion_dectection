{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7d57af32-b700-4d98-bee3-47dd5c51d33e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🧹 Cleaning images in 'angry'...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3995/3995 [00:38<00:00, 103.08it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🧹 Cleaning images in 'disgust'...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 436/436 [00:03<00:00, 122.06it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🧹 Cleaning images in 'fear'...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4097/4097 [00:28<00:00, 142.63it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🧹 Cleaning images in 'happy'...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3790/3790 [00:03<00:00, 1162.51it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "✅ Data cleaning complete! Cleaned images saved to 'data_cleaned/train'\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "# emotion_data_cleaning.py\n",
    "import os\n",
    "import cv2\n",
    "from tqdm import tqdm\n",
    "\n",
    "# Define dataset path\n",
    "data_dir = \"data/train\"   # adjust if needed\n",
    "output_dir = \"data_cleaned/train\"\n",
    "\n",
    "# Create the cleaned data folder if it doesn't exist\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "\n",
    "# Loop through each emotion subfolder\n",
    "for emotion in os.listdir(data_dir):\n",
    "    emotion_path = os.path.join(data_dir, emotion)\n",
    "    cleaned_emotion_path = os.path.join(output_dir, emotion)\n",
    "    os.makedirs(cleaned_emotion_path, exist_ok=True)\n",
    "\n",
    "    print(f\"🧹 Cleaning images in '{emotion}'...\")\n",
    "\n",
    "    for img_file in tqdm(os.listdir(emotion_path)):\n",
    "        img_path = os.path.join(emotion_path, img_file)\n",
    "        try:\n",
    "            img = cv2.imread(img_path)\n",
    "            if img is None:\n",
    "                continue  # skip invalid images\n",
    "\n",
    "            # Convert to grayscale and resize\n",
    "            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "            resized = cv2.resize(gray, (48, 48))\n",
    "\n",
    "            save_path = os.path.join(cleaned_emotion_path, img_file)\n",
    "            cv2.imwrite(save_path, resized)\n",
    "        except Exception as e:\n",
    "            print(f\"Error processing {img_file}: {e}\")\n",
    "\n",
    "print(\"\\n✅ Data cleaning complete! Cleaned images saved to 'data_cleaned/train'\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f95b871e-5fcb-4dac-9a8c-60b325e4f1a6",
   "metadata": {},
   "outputs": [],
   "source": [
    " "
   ]
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
