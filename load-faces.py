import os
import cv2
import numpy as np

ANNOTATIONS_PATH = os.path.join('.', 'data', 'train_annotations', 'wider_face_train_bbx_gt.txt')

def load_annotations():
    annotations = {}
    with open(ANNOTATIONS_PATH, 'r') as f:
        lines = f.readlines()
        current_image = None
        num_annotations = 0
        annotation_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.endswith('.jpg'):
                current_image = os.path.splitext(os.path.basename(line.strip()))[0]
                annotations[current_image] = []
                annotation_count = 0
            elif current_image and annotation_count == 0:
                try:
                    num_annotations = int(line)
                    annotation_count = num_annotations
                except ValueError:
                    print(f"Invalid integer value found for number of annotations in line: {line}")
                    continue
            else:
                if current_image and annotation_count > 0:
                    try:
                        annotation = [float(x) for x in line.split()]
                        annotations[current_image].append(annotation)
                        annotation_count -= 1
                    except ValueError:
                        print(f"Invalid float value found in annotation line: {line}")
                        continue
        # Check if there are any remaining lines in the file
        if lines:
            print(f"Warning: {len(lines)} lines remaining in the file.")

    return annotations

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to normalize lighting
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Sobel filter to enhance edges
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.sqrt(sobelx**2 + sobely**2)
    
    # Convert Sobel result back to uint8
    sobel = cv2.convertScaleAbs(sobel)
    
    return sobel

def load_images_with_annotations(dataset_path, annotations):
    images = []
    annotations_list = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Extract the image name from the file path
                image_name = os.path.splitext(file)[0]
                
                if image_name in annotations:
                    images.append(processed_image)
                    annotations_list.append(annotations[image_name])
    
    return images, annotations_list

# Load annotations once
annotations = load_annotations()
for key, value in annotations.items():
    print(f"Sample annotation loaded: {key}: {value}")
    break

# Load images and their corresponding annotations
dataset_path = './data/train'
images, annotations_list = load_images_with_annotations(dataset_path, annotations)

print(f'Loaded {len(images)} images and {len(annotations_list)} annotations.')

# Print a sample annotation for verification
if images and annotations_list:
    print('Sample image shape: ', images[0].shape)
    print('Sample annotations: ', annotations_list[0])
    
cv2.imshow('Filtered Image', images[0])
cv2.waitKey(0) 
cv2.destroyAllWindows()  
