import streamlit as st
import cv2
import os
import pandas as pd
import numpy as np
import datetime
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
        df = pd.DataFrame(columns=["Enrollment", "Name"])
        df.to_csv(STUDENT_DETAIL_PATH, index=False)

    df = pd.read_csv(STUDENT_DETAIL_PATH)
    df.loc[len(df)] = [enrollment, name]
    df.to_csv(STUDENT_DETAIL_PATH, index=False)


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

    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINING_LABEL_PATH)


def take_attendance(subject):
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

    if st.button("📸 Capture Face"):
        if enrollment and name:
            capture_face(enrollment, name)
            st.success(f"✅ Face registered for {name} ({enrollment})")
        else:
            st.warning("⚠️ Please enter Enrollment No and Name.")

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
            df = pd.read_csv(file_path)
            st.dataframe(df)
        else:
            st.info("No attendance files found for this subject.")
    else:
        st.info("No subjects found. Take attendance first.")
