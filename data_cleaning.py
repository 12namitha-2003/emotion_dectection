{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "39caf509-f4ee-423a-a2de-1a5164e51f65",
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
      "100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3995/3995 [00:39<00:00, 101.73it/s]\n"
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
      "100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 436/436 [00:04<00:00, 107.29it/s]\n"
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
      "100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4097/4097 [00:39<00:00, 103.42it/s]\n"
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
      "100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3790/3790 [00:37<00:00, 101.14it/s]\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import cv2\n",
    "from tqdm import tqdm\n",
    "\n",
    "def clean_dataset(input_dir, output_dir):\n",
    "    os.makedirs(output_dir, exist_ok=True)\n",
    "\n",
    "    for emotion in os.listdir(input_dir):\n",
    "        emotion_path = os.path.join(input_dir, emotion)\n",
    "        cleaned_emotion_path = os.path.join(output_dir, emotion)\n",
    "        os.makedirs(cleaned_emotion_path, exist_ok=True)\n",
    "\n",
    "        print(f\"🧹 Cleaning images in '{emotion}'...\")\n",
    "        for img_file in tqdm(os.listdir(emotion_path)):\n",
    "            img_path = os.path.join(emotion_path, img_file)\n",
    "            try:\n",
    "                img = cv2.imread(img_path)\n",
    "                if img is None:\n",
    "                    continue\n",
    "                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "                resized = cv2.resize(gray, (48, 48))\n",
    "                save_path = os.path.join(cleaned_emotion_path, img_file)\n",
    "                cv2.imwrite(save_path, resized)\n",
    "            except Exception as e:\n",
    "                print(f\"Error with {img_file}: {e}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    # Clean both train and test data\n",
    "    clean_dataset(\"data/train\", \"data_cleaned/train\")\n",
    "    clean_dataset(\"data/test\", \"data_cleaned/test\")\n",
    "    print(\"\\n✅ Dataset cleaning complete! Files saved in 'data_cleaned/' folder.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0fc68df8-11a2-4a9e-b166-1b3af92e5a66",
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
