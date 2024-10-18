import os
import cv2

def extract_and_sample_frames(video_path, output_folder, sample_rate=5): #extract every 5th frame as huge dataset
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    sampled_frame_count = 0
    
    # Check if the video was opened correctly
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Sample every nth frame based on the sample_rate
        if frame_count % sample_rate == 0:
            # Construct the filename and save the frame
            frame_filename = f"{video_name}_frame_{sampled_frame_count}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            sampled_frame_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"Extracted and sampled {sampled_frame_count} frames from {video_name}")

# Example usage
video_folder = 'Resized_Videos'  # Folder containing video files
output_folder = 'Extracted_Sampled_Frames'  # Folder to save extracted frames
sample_rate = 3  # Extract every 3rd frame

# Extract and sample frames from all videos in the folder
for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    extract_and_sample_frames(video_path, output_folder, sample_rate=sample_rate)
