import streamlit as st
import cv2
import os
import pandas as pd
import numpy as np
import datetime
from twilio.rest import Client
from PIL import Image

# ------------------------------
# PATHS & FOLDERS
# ------------------------------
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINING_IMAGE_PATH = "TrainingImage"
TRAINING_LABEL_PATH = "TrainingImageLabel/Trainner.yml"
STUDENT_DETAIL_PATH = "StudentDetails/studentdetails.csv"
ATTENDANCE_PATH = "Attendance"

os.makedirs(TRAINING_IMAGE_PATH, exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs(ATTENDANCE_PATH, exist_ok=True)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def capture_face(enrollment, name):
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    sample_count = 0


    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw camera cut-out (central circle)
        h_img, w_img = img.shape[:2]
        center = (w_img // 2, h_img // 2)
        radius = min(w_img, h_img) // 4
        cv2.circle(img, center, radius, (0,255,255), 2)

        for (x, y, w, h) in faces:
            sample_count += 1
            cv2.imwrite(f"{TRAINING_IMAGE_PATH}/{name}.{enrollment}.{sample_count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Capturing Faces", img)
        if cv2.waitKey(100) & 0xFF == 27:  # ESC
            break
        elif sample_count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()

    if not os.path.exists(STUDENT_DETAIL_PATH):
        df = pd.DataFrame(columns=["Enrollment", "Name", "Phone", "VoicePath"])
        df.to_csv(STUDENT_DETAIL_PATH, index=False)

    df = pd.read_csv(STUDENT_DETAIL_PATH)
    # Phone and VoicePath will be added in the registration UI logic
    # This function only handles face capture
    # The actual row will be added after all data is collected
    pass


def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []
    image_paths = [os.path.join(TRAINING_IMAGE_PATH, f) for f in os.listdir(TRAINING_IMAGE_PATH)]
    
    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        parts = os.path.split(image_path)[-1].split(".")
        if len(parts) > 2 and parts[1].isdigit():
            id = int(parts[1])
            faces.append(gray_img)
            ids.append(id)
    if len(faces) < 2:
        st.error("You need at least two face samples to train the model. Please register more students or capture more images.")
        return
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINING_LABEL_PATH)


def take_attendance(subject):
    if not os.path.exists(TRAINING_LABEL_PATH):
        st.error("Model not trained! Please train the model before taking attendance.")
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINING_LABEL_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    df = pd.read_csv(STUDENT_DETAIL_PATH)
    cam = cv2.VideoCapture(0)
    attendance = pd.DataFrame(columns=["Enrollment", "Name", "Date", "Time"])
    future = datetime.datetime.now() + datetime.timedelta(seconds=20)

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                matched = df.loc[df["Enrollment"] == id]
                if not matched.empty:
                    name = matched["Name"].values[0]
                else:
                    name = f"Unknown_{id}"
                # If there are twins or confidence is low, prompt for voice verification (placeholder)
                if conf > 50:  # Lower confidence, possible ambiguity
                    # Placeholder: In a real app, prompt for voice and compare with registered sample
                    print(f"[VOICE VERIFICATION NEEDED] for {name} (ID: {id}) - confidence: {conf}")
                    # You can add Streamlit audio input and compare here
                now = datetime.datetime.now()
                attendance.loc[len(attendance)] = [id, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]
                cv2.putText(img, f"{name}-{id}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            else:
                cv2.putText(img, "Unknown", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

        cv2.imshow("Attendance", img)
        if datetime.datetime.now() > future:
            break
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


    os.makedirs(f"{ATTENDANCE_PATH}/{subject}", exist_ok=True)
    filename = f"{ATTENDANCE_PATH}/{subject}/{subject}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    attendance.drop_duplicates(subset=["Enrollment"], inplace=True)
    attendance.to_csv(filename, index=False)

    # --- SMS Notification for Absentees ---
    # Twilio setup (replace with your credentials)
    TWILIO_ACCOUNT_SID = "your_account_sid"
    TWILIO_AUTH_TOKEN = "your_auth_token"
    TWILIO_PHONE = "+1234567890"  # Your Twilio phone number
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    # Load all registered students
    df_students = pd.read_csv(STUDENT_DETAIL_PATH)
    present_ids = set(attendance["Enrollment"].astype(str))
    for idx, row in df_students.iterrows():
        if str(row["Enrollment"]) not in present_ids:
            phone = str(row["Phone"])
            name = row["Name"]
            message = f"Your ward {name} is absent for the class. Please contact your ward for a reason."
            try:
                client.messages.create(
                    body=message,
                    from_=TWILIO_PHONE,
                    to=phone
                )
            except Exception as e:
                print(f"Failed to send SMS to {phone}: {e}")

    return filename


# ------------------------------
# STREAMLIT APP UI
# ------------------------------
st.set_page_config(page_title="Class Vision", page_icon="📚", layout="centered")

# CSS styling to look like your Tkinter dark theme
st.markdown("""
    <style>
        body {
            background-color: #1c1c1c;
            color: yellow;
        }
        .stButton button {
            background-color: black;
            color: yellow;
            border-radius: 10px;
            font-size: 18px;
            height: 3em;
            width: 15em;
        }
        .stButton button:hover {
            background-color: yellow;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📚 CLASS VISION")
st.markdown("### Welcome to the Automated Attendance System")

menu = ["🏫 Register Student", "⚙️ Train Model", "📝 Take Attendance", "👨‍🎓 View Students", "📊 View Attendance"]
choice = st.sidebar.radio("Menu", menu)


# ------------------------------
# MENU FUNCTIONS
# ------------------------------
if choice == "🏫 Register Student":
    st.subheader("Register a New Student")
    enrollment = st.text_input("Enrollment No")
    name = st.text_input("Student Name")
    phone = st.text_input("Parent Phone Number")
    voice = st.file_uploader("Upload student voice sample (.wav)", type=["wav"])

    if st.button("📸 Capture Face & Register"):
        if enrollment and name and phone and voice is not None:
            # Save face images
            capture_face(enrollment, name)
            # Save voice sample
            voice_dir = "StudentDetails/voices"
            os.makedirs(voice_dir, exist_ok=True)
            voice_path = f"{voice_dir}/{enrollment}_{name}.wav"
            with open(voice_path, "wb") as f:
                f.write(voice.read())
            # Update CSV
            if not os.path.exists(STUDENT_DETAIL_PATH):
                df = pd.DataFrame(columns=["Enrollment", "Name", "Phone", "VoicePath"])
                df.to_csv(STUDENT_DETAIL_PATH, index=False)
            df = pd.read_csv(STUDENT_DETAIL_PATH)
            df.loc[len(df)] = [enrollment, name, phone, voice_path]
            df.to_csv(STUDENT_DETAIL_PATH, index=False)
            st.success(f"✅ Registered {name} ({enrollment}) with phone and voice sample.")
        else:
            st.warning("⚠️ Please enter all details and upload a voice sample.")

elif choice == "⚙️ Train Model":
    st.subheader("Train the Face Recognition Model")
    if st.button("🧠 Train Model"):
        train_model()
        st.success("✅ Model trained successfully!")

elif choice == "📝 Take Attendance":
    st.subheader("Take Attendance")
    subject = st.text_input("Enter Subject Name")
    if st.button("🎥 Start Attendance"):
        if subject:
            file = take_attendance(subject)
            st.success(f"✅ Attendance saved: {file}")
        else:
            st.warning("⚠️ Please enter a subject name.")


elif choice == "👨‍🎓 View Students":
    st.subheader("Registered Students")
    if os.path.exists(STUDENT_DETAIL_PATH):
        df = pd.read_csv(STUDENT_DETAIL_PATH)
        st.dataframe(df)
    else:
        st.warning("⚠️ No student records found yet.")

# ------------------------------
# VIEW ATTENDANCE FEATURE
# ------------------------------
elif choice == "📊 View Attendance":
    st.subheader("View Attendance Records")
    subjects = []
    if os.path.exists(ATTENDANCE_PATH):
        subjects = [d for d in os.listdir(ATTENDANCE_PATH) if os.path.isdir(os.path.join(ATTENDANCE_PATH, d))]
    selected_subject = st.selectbox("Select Subject", subjects)
    if selected_subject:
        attendance_files = [f for f in os.listdir(os.path.join(ATTENDANCE_PATH, selected_subject)) if f.endswith('.csv')]
        selected_file = st.selectbox("Select Attendance File", attendance_files)
        if selected_file:
            file_path = os.path.join(ATTENDANCE_PATH, selected_subject, selected_file)
            df_attendance = pd.read_csv(file_path)
            st.dataframe(df_attendance)
            # Show summary stats and absentees
            if os.path.exists(STUDENT_DETAIL_PATH):
                df_students = pd.read_csv(STUDENT_DETAIL_PATH)
                total_students = len(df_students)
                present_ids = set(df_attendance["Enrollment"].astype(str))
                all_ids = set(df_students["Enrollment"].astype(str))
                absentees = df_students[~df_students["Enrollment"].astype(str).isin(present_ids)]
                st.markdown(f"**Total Students:** {total_students}")
                st.markdown(f"**Present:** {len(present_ids)}")
                st.markdown(f"**Absent:** {total_students - len(present_ids)}")
                st.markdown("**Absentees List:**")
                if not absentees.empty:
                    display_cols = [col for col in ["Enrollment", "Name", "Phone"] if col in absentees.columns]
                    st.dataframe(absentees[display_cols])
                else:
                    st.info("No absentees!")
        else:
            st.info("No attendance files found for this subject.")
    else:
        st.info("No subjects found. Take attendance first.")
