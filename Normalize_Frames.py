import os
import numpy as np
import cv2

def normalize_and_save_frames(frames_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all frames in the frames folder
    for frame_name in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: {frame_path} could not be read.")
            continue
        
        # Normalize frame
        normalized_frame = frame / 255.0  # Normalize to [0, 1]
        
        # Convert normalized frame back to 8-bit for saving
        normalized_frame = (normalized_frame * 255).astype(np.uint8)

        # Save the normalized frame as an image
        output_frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_frame_path, normalized_frame)  # Save as image file

    print("All frames have been normalized and saved.")

if __name__ == "__main__":
    frames_folder = 'Extracted_Sampled_Frames'  # Folder where frames are stored
    output_folder = 'Normalized_Frames'  # Folder to save normalized frames

    normalize_and_save_frames(frames_folder, output_folder)
