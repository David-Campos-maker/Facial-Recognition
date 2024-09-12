# Import necessary libraries
import cv2
import dlib

# Function to detect faces in a video stream
def detect_faces():
    # Initialize the face detector
    detector = dlib.get_frontal_face_detector()
    
    # Initialize the predictor to detect facial landmarks
    predictor = dlib.shape_predictor('./haarcascades/shape_predictor_68_face_landmarks.dat')
    
    # Open the webcam
    capture = cv2.VideoCapture(0)
    
    # Loop until the user presses 'q' to quit
    while True:
        # Read a frame from the webcam
        ret, frame = capture.read()
        
        # If the frame is not read correctly, break the loop
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = detector(gray)
        
        # Loop through each detected face
        for face in faces:
            # Get the coordinates of the face bounding box
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Detect facial landmarks in the grayscale frame
            landmarks = predictor(gray, face)
            
            # Loop through each facial landmark and draw a circle
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Display the frame with detected faces
        cv2.imshow('Faces detected', frame)
        
        # If the user presses 'q', break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and destroy all OpenCV windows
    capture.release()
    cv2.destroyAllWindows()

# Run the face detection program
if __name__ == '__main__':
    detect_faces()
