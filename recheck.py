import os

train_path = "data_cleaned/train"

for emotion in os.listdir(train_path):
    path = os.path.join(train_path, emotion)
    if os.path.isdir(path):
        print(f"{emotion}: {len(os.listdir(path))} images")
