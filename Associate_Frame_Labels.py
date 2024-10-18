import pandas as pd
import os

def associate_frame_labels(video_labels_file, frames_folder, output_file):
    # Load video labels
    labels_df = pd.read_csv(video_labels_file)
    
    # Create a list to store frame label associations
    frame_labels = []

    # Loop through each video label
    for index, row in labels_df.iterrows():
        video_name = os.path.splitext(row['Video Name'])[0]  # Remove .mp4 extension
        label = row['Label']
        
        # Find all frames associated with the video
        for frame in os.listdir(frames_folder):
            if video_name in frame:  # Check if frame belongs to the video
                frame_labels.append({'Frame Name': frame, 'Label': label})

    # Create a DataFrame and save it to CSV
    frame_labels_df = pd.DataFrame(frame_labels)
    frame_labels_df.to_csv(output_file, index=False)

# Example usage
video_labels_file = 'Labels.csv'  # Path to your video labels CSV
frames_folder = 'Extracted_Sampled_Frames'  # Folder where frames are stored
output_file = 'Frame_Labels.csv'  # Output CSV for frame labels

associate_frame_labels(video_labels_file, frames_folder, output_file)
