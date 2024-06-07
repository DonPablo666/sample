import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tflite_runtime.interpreter import Interpreter
import time
import vonage

# Vonage (Nexmo) configuration
# Vonage (Nexmo) configuration
client = vonage.Client(key="635c07e3", secret="G3HeAtyp2N8M2ANP")
sms = vonage.Sms(client)
from_phone_number = 'Vonage APIs'  # The sender ID or phone number
to_phone_number = '639054256224'  # Replace with the recipient's phone number

# Load the TFLite model
model_path = '/home/pi/env/yoloapp/yoloapp/model.tflite'
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

img_height, img_width = input_shape[1], input_shape[2]

# Class names (adjust these to match your model's classes)
class_names = ["Bacterial", "fungal", "healthy"]

# Function to preprocess images
def preprocess_image(image):
    image = cv2.resize(image, (img_height, img_width))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32)  # Ensure the data type is float32
    return image

# Function to send email
def send_email(subject, body, to_email):
    from_email = "smartiotproject1@gmail.com"  # Replace with your email
    from_password = "ofbc jbky zmte mdpy"  # Replace with your app-specific password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to send SMS via Vonage (Nexmo)
def send_sms(body):
    try:
        response_data = vonage_sms.send_message({
            "from": from_phone_number,
            "to": to_phone_number,
            "text": body,
        })
        if response_data["messages"][0]["status"] == "0":
            print("SMS sent successfully")
        else:
            print(f"Failed to send SMS: {response_data['messages'][0]['error-text']}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Variable to store the timestamp of the last alert sent
last_alert_time = 0
alert_interval = 3600  # 1 hour in seconds

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_image(frame)

    # Set the tensor to the frame
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data[0])

    # Display the result
    cv2.putText(frame, f'Pred: {class_names[prediction]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Check if the prediction is Bacterial or fungal and send an email and SMS
    current_time = time.time()
    if class_names[prediction] in ["Bacterial", "fungal"] and (current_time - last_alert_time > alert_interval):
        subject = "Alert: Lettuce Health Issue Detected"
        body = f"A {class_names[prediction]} infection has been detected in the lettuce. The lettuce is not healthy."
        to_email = "smartiotproject1@gmail.com"

        send_email(subject, body, to_email)
        send_sms(body)

        last_alert_time = current_time  # Update the last alert time

    # Show the frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
