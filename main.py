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

PATH_TO_LABELS = "everything_labels.txt"
PATH_TO_MODEL = "everything_model.tflite"

GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')

last_email_time = 0

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def send_email(subject, body):
    sender_email = "josephmorrison702@gmail.com"
    receiver_email = "josephmorrison702@gmail.com"
    password = GMAIL_PASSWORD

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(sender_email, password)
        
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.close()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

interpreter = tf.lite.Interpreter(model_path=PATH_TO_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    count = interpreter.get_tensor(output_details[3]['index'])[0]

    detectable_objects = ["cat", "dog"]

    for i in range(int(count)):
        if scores[i] > 0.75:
            object_name = labels[int(classes[i])]
            if object_name in detectable_objects:
                current_time = time.time()
                if current_time - last_email_time >= 30:
                    send_email("Pet Detected", f"Hello,\n\nYour {object_name} has been detected with {scores[i]*100:.2f}% confidence.\n\nSincerely,\nSmart Pet Detection")
                    print("Email sent successfully!")
                    last_email_time = current_time
                else:
                    print(f"Email not sent. {30-(current_time - last_email_time):.2f} more seconds.")

    cv2.imshow('Object Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
