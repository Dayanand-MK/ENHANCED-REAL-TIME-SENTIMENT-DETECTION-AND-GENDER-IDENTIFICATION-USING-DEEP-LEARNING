import os
import pandas as pd
from sklearn.model_selection import train_test_split

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]

ck_dataset_dir = "E:\Mini Project\Video Datasset"

image_paths = []
labels = []

for emotion_index, emotion in enumerate(emotions):
    emotion_folder = os.path.join(ck_dataset_dir, emotion)
    for image in os.listdir(emotion_folder):
        image_path = os.path.join(emotion_folder, image)
        image_paths.append(image_path)
        labels.append(emotion_index)  

df = pd.DataFrame({"image_path": image_paths, "label": labels})

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_df.to_csv("Train1.csv", index=False)
test_df.to_csv("Test_1.csv", index=False)

print("CSV files for training and testing created successfully!")