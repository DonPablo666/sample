from ultralytics import YOLO
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import vonage

# Email configuration
from_email = "your_email@gmail.com"
from_password = "your_app_specific_password"
to_email = "recipient_email@gmail.com"

# Vonage (Nexmo) configuration
client = vonage.Client(key="your_vonage_key", secret="your_vonage_secret")
sms = vonage.Sms(client)
from_phone_number = 'Vonage APIs'  # The sender ID or phone number
to_phone_number = 'recipient_phone_number'  # Replace with the recipient's phone number

# Function to send email
def send_email(subject, body, to_email):
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
        response_data = sms.send_message({
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

# Load the YOLO model
model = YOLO(r"C:\Users\ACER TRAVEL  MATE\Downloads\yolov8\best.pt")

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform the detection
    results = model.predict(frame, show=False, conf=0.5)
    
    # Draw the results on the frame
    annotated_frame = results[0].plot()

    # Check for detected diseases
    detected_diseases = []
    for result in results[0].boxes.data:
        label = result[0]
        if label in ['Bacterial', 'Downy mildew', 'Septoria Blight', 'Viral', 'Wilt and leaf blight', 'powdery mildew']:  # Adjust the labels to match your model's output
            detected_diseases.append(label)
    
    if detected_diseases:
        subject = "Alert: Disease Detected"
        body = f"Diseases detected: {', '.join(detected_diseases)}"
        
        # Send email and SMS
        send_email(subject, body, to_email)
        send_sms(body)
    
    # Display the frame
    cv2.imshow('YOLO Detection', annotated_frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
