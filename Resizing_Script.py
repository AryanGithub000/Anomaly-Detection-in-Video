#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os

def resize_video(input_video_path, output_video_path, frame_size=(224, 224)):
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the video's FPS (frames per second)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get video codec information and set up VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, frame_size)
        
        # Write the resized frame to output video
        out.write(frame)
    
    cap.release()
    out.release()

def resize_all_videos(input_folder, output_folder, frame_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for video_name in os.listdir(input_folder):
        if video_name.endswith(('.mp4', '.mov', '.avi')):
            input_video_path = os.path.join(input_folder, video_name)
            output_video_path = os.path.join(output_folder, video_name)
            
            print(f"Resizing {video_name}...")
            resize_video(input_video_path, output_video_path, frame_size)
            print(f"Resized {video_name} saved to {output_video_path}.")

# Define input and output folders
input_folder = 'CCTV Data'  # Change to your actual input folder
output_folder = 'Resized_Videos'  # Output folder where resized videos will be saved

# Resize all videos
resize_all_videos(input_folder, output_folder)


# In[ ]:




