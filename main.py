import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

load_dotenv()

# Path to label map file and TFLite model
PATH_TO_LABELS = "everything_labels.txt"
PATH_TO_MODEL = "everything_model.tflite"

GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')

last_email_time = 0

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def send_email(subject, body):
    # Your email credentials
    sender_email = "josephmorrison702@gmail.com"
    receiver_email = "josephmorrison702@gmail.com"
    password = GMAIL_PASSWORD

    # Creating the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connecting to the Gmail SMTP server
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(sender_email, password)
        
        # Sending the email and closing the connection
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.close()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=PATH_TO_MODEL)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

# Start capturing video from the built-in camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess frame for model input
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Feed the frame to the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # You might need to adjust these based on how your model outputs data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    count = interpreter.get_tensor(output_details[3]['index'])[0]  # Number of detections

    detectable_objects = ["cat", "dog"]

    for i in range(int(count)):
        if scores[i] > 0.75:
            object_name = labels[int(classes[i])]
            if object_name in detectable_objects:
                current_time = time.time()  # Get the current time
                # Check if 30 seconds have elapsed since the last email was sent
                if current_time - last_email_time >= 30:
                    # Send the email
                    send_email("Pet Detected", f"Hello,\n\nYour {object_name} has been detected with {scores[i]*100:.2f}% confidence.\n\nSincerely,\nSmart Pet Detection")
                    # Update the last email time to the current time
                    print("Email sent successfully!")
                    last_email_time = current_time
                else:
                    print(f"Email not sent. {30-(current_time - last_email_time):.2f} more seconds.")

    # Display the resulting frame
    cv2.imshow('Object Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
