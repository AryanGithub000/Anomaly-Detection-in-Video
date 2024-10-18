import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
frames_folder = 'Grayscale_Frames'  # Folder containing all grayscale frames
labels_file = 'Frame_Labels.csv'     # CSV file with frame labels
output_folder = 'Data_Split'         # Folder to save the train/val/test splits

# Create directories for train, val, and test sets
train_folder = os.path.join(output_folder, 'train')
val_folder = os.path.join(output_folder, 'val')
test_folder = os.path.join(output_folder, 'test')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Load the labels CSV
labels_df = pd.read_csv(labels_file)

# Split the data
train_df, temp_df = train_test_split(labels_df, test_size=0.3, random_state=42, stratify=labels_df['Label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Label'])

def copy_frames(frame_df, destination_folder):
    for index, row in frame_df.iterrows():
        frame_name = row['Frame Name']
        label = row['Label']
        src_path = os.path.join(frames_folder, frame_name)
        label_folder = os.path.join(destination_folder, str(label))  # Create subfolder by label (0, 1)
        os.makedirs(label_folder, exist_ok=True)
        dst_path = os.path.join(label_folder, frame_name)
        shutil.copy(src_path, dst_path)

# Copy frames to respective folders
copy_frames(train_df, train_folder)
copy_frames(val_df, val_folder)
copy_frames(test_df, test_folder)

print(f"Dataset split done. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
