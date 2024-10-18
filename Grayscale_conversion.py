import cv2
import os

def convert_to_grayscale(frames_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for frame_file in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_file)
        
        # Load the frame
        img = cv2.imread(frame_path)
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save the grayscale frame
        output_frame_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_frame_path, img_gray)
    
    print(f"Frames converted to grayscale and saved in {output_folder}")

# Example usage
frames_folder = 'Normalized_Frames'
output_folder = 'Grayscale_Frames'
convert_to_grayscale(frames_folder, output_folder)
