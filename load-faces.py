import os
import cv2
import numpy as np

ANNOTATIONS_PATH = os.path.join('.', 'data', 'train_annotations', 'wider_face_train_bbx_gt.txt')

def load_annotations():
    """
    Load annotations from the specified file and return them as a dictionary.

    The annotations file is expected to have the following format:
    - Each line represents an image and its annotations.
    - The first line of each image entry is the image file name (without extension).
    - The second line of each image entry is the number of annotations for that image.
    - The subsequent lines of each image entry are the annotations, each represented as a space-separated list of floats.

    Parameters:
    None

    Returns:
    dict: A dictionary where keys are image names and values are lists of annotations.

    Raises:
    IOError: If the annotations file cannot be opened or read.
    ValueError: If an invalid integer or float value is found in the annotations file.
    """
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
                        annotation = tuple(map(float, line.split()))
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
    """
    Preprocess an input image by converting it to grayscale, applying Gaussian blur,
    and then applying a Sobel filter to enhance edges.

    Parameters:
    image (numpy.ndarray): The input image in BGR format.

    Returns:
    numpy.ndarray: The preprocessed image in grayscale format with enhanced edges.
    """

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
    """
    Load images and their corresponding annotations from a dataset directory.

    Parameters:
    dataset_path (str): The path to the dataset directory containing the images.
    annotations (dict): A dictionary where keys are image names and values are lists of annotations.

    Returns:
    tuple: A tuple containing two dictionaries. The first dictionary contains the loaded images,
           with keys as image names and values as processed images.
           The second dictionary contains the annotations, with keys as image names and values as lists of annotations.
    """
    images = {}
    annotations_dict = {}

    # Walk through the dataset directory
    for root, _, files in os.walk(dataset_path):
        for file in files:
            # Check if the file is a JPEG image
            if file.endswith('.jpg'):
                # Construct the full path to the image file
                image_path = os.path.join(root, file)
                
                # Read the image using OpenCV
                image = cv2.imread(image_path)
                
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Extract the image name from the file path
                image_name = os.path.splitext(file)[0]
                
                # Check if the image has annotations
                if image_name in annotations:
                    # Add the image and its annotations to the respective dictionaries
                    images[image_name] = processed_image
                    annotations_dict[image_name] = annotations[image_name]
    
    # Return the loaded images and annotations
    return images, annotations_dict

# Load annotations once
annotations = load_annotations()
for key, value in annotations.items():
    print(f"Sample annotation loaded: {key}: {value}")
    break

# Load images and their corresponding annotations
dataset_path = './data/train'
images, annotations_dict = load_images_with_annotations(dataset_path, annotations)

print(f'Loaded {len(images)} images and {len(annotations_dict)} annotations.')

image_name = '0_Parade_marchingband_1_849'

if image_name in images and image_name in annotations_dict:
    print(f'Image Name: {image_name}')
    print(f'Image Shape: {images[image_name].shape}')
    print(f'Annotations: {annotations_dict[image_name]}')
else:
    print(f'Image {image_name} not found in the dataset.')
    
cv2.imshow('Filtered Image - 0_Parade_marchingband_1_849', images[image_name])
cv2.waitKey(0) 
cv2.destroyAllWindows()