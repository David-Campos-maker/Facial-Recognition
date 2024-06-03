import os
import cv2
import numpy as np

ANNOTATIONS_PATH = os.path.join('.', 'data', 'train_annotations', 'wider_face_train_bbx_gt.txt')

def load_annotations(annotations_path):
    annotations = {}
    with open(annotations_path, 'r') as f:
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
                    # Handle case where the line cannot be converted to an integer
                    pass
            else:
                if current_image and annotation_count > 0:
                    try:
                        annotation = [float(x) for x in line.split()]
                        annotations[current_image].append(annotation)
                        annotation_count -= 1
                    except ValueError:
                        # Handle case where the line cannot be converted to floats
                        pass
    return annotations

def load_images_with_annotations(dataset_path, annotations):
    images = []
    annotations_list = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                
                # Extract the image name from the file path
                image_name = os.path.splitext(file)[0]
                
                if image_name in annotations:
                    images.append(image)
                    # Flatten the list of annotations for this image
                    images_annotations = [item for sublist in annotations[image_name] for item in sublist]
                    annotations_list.append(images_annotations)
    
    return images, annotations_list

# Load annotations once
annotations = load_annotations(ANNOTATIONS_PATH)

# Load images and their corresponding annotations
dataset_path = './data/train'
images, annotations_list = load_images_with_annotations(dataset_path, annotations)

print(f'Loaded {len(images)} images and {len(annotations_list)} annotations.')

# Print a sample annotation for verification
if images and annotations_list:
    print('Sample image shape:', images[0].shape)
    print('Sample annotations:', annotations_list[0])
