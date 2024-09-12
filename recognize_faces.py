# Import necessary libraries
import cv2
import dlib
import pickle
import numpy as np
import requests
import tkinter as tk
from tkinter import messagebox

# Initialize the face detector, predictor, and face recognizer
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./haarcascades/shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('./haarcascades/dlib_face_recognition_resnet_model_v1.dat')

# Load the face descriptors, labels, and label dictionary from the pickle file
with open('face_descriptors.pkl', 'rb') as f:
    descriptors, labels, label_dict = pickle.load(f)

# Function to preprocess the image before face recognition
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    return gray

# Function to recognize the face in the image
def recognize_face(descriptor, descriptors, labels, threshold=0.60, confidence_threshold=0.45):
    distances = np.linalg.norm(descriptors - descriptor, axis=1)
    min_distance = np.min(distances)
    confidence = 1 - min_distance
    
    if min_distance < threshold and confidence > confidence_threshold:
        return labels[np.argmin(distances)], min_distance
    
    return -1, min_distance

# Function to get the device's location using IP address
def get_location():
    try:
        # Send a GET request to ipinfo.io to get the device's IP address and location
        response = requests.get('https://ipinfo.io')
        data = response.json()
        ip = data['ip']
        loc = data['loc'].split(',')
        latitude = float(loc[0])
        longitude = float(loc[1])
        
        return ip, latitude, longitude
    
    except requests.RequestException as e:
        # Print the error message if there is an exception while sending the GET request
        print(f"Error getting location: {e}")
        return None, None, None

# Function to show an alert message with the recognized face's details
def show_alert(name, x, y, latitude, longitude):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # Display the alert message with the recognized face's details
    if latitude and longitude:
        message = f"Recognized face: {name}\nPosition: ({x}, {y})\nLocation: ({latitude}, {longitude})"
    else:
        message = f"Recognized face: {name}\nPosition: ({x}, {y})\nLocation: Not available"
        
    messagebox.showinfo("Facial Recognition Alert", message)
    root.destroy()  # Destroy the root window

# Function to recognize faces in a video stream
def recognize_faces():
    capture = cv2.VideoCapture(0)
    
    # Get the device's location using IP address
    ip, latitude, longitude = get_location()  

    while True:
        ret, frame = capture.read()
        
        if not ret:
            break
        
        gray = preprocess_image(frame)
        faces = detector(gray)
        
        for face in faces:
            shape = predictor(gray, face)
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
            label, confidence = recognize_face(face_descriptor, np.array(descriptors), np.array(labels))
            name = label_dict.get(label, "Unknown")
            
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'{name} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if name!= "Unknown":
                show_alert(name, x, y, latitude, longitude)
            
        cv2.imshow('Recognized Faces', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    capture.release()
    cv2.destroyAllWindows()

# Run the face recognition program
if __name__ == '__main__':
    recognize_faces()