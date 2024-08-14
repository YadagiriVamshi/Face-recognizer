
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator 
import regression
import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np
import mysql.connector

def train_classifier():
    data_dir = "C:/Users/NEELAM VAMSHI KRISHN/OneDrive/Desktop/face recognizer/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training dataset completed!!!')

def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for(x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            id, pred = clf.predict(gray_image[y:y+h, x:x+w])
            confidence = int(100*(1-pred/300))

            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="",
                database="Authorized_user"
            )
            mycursor = mydb.cursor()
            mycursor.execute("select * from my_table where id="+str(id))
            row = mycursor.fetchone()
            if row is not None:
                name, age, address = row[1], row[2], row[3]
                text = f"Name: {name}, Age: {age}, Address: {address}"

            if confidence > 80:
                cv2.putText(img, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (235, 212 , 7), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face detection", img)

        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
    else:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="Authorized_user"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * from my_table")
        myresult = mycursor.fetchall()
        id = 1
        for x in myresult:
            id += 1
        sql = "insert into my_table(id,Name,Age,Address) values(%s,%s,%s,%s)"
        val = (id, t1.get(), t2.get(), t3.get())
        mycursor.execute(sql, val)
        mydb.commit()

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return None
            for(x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Cropped face", face)
                if cv2.waitKey(1) == 13 or int(img_id) == 200:
                    break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!!!')

def center_align(widget):
    widget.place(relx=0.5, rely=0.5, anchor='center')

window = tk.Tk()
window.title("Face2AI-Face Recognizer by ZT-16")
window.config(background="lightblue")
window.geometry("800x600")

title_font = ("Arial", 24)
label_font = ("Arial", 16)

title_label = tk.Label(window, text="Face2AI-Face Recognizer", font=title_font, bg="lightblue")
title_label.pack(pady=20)

t1_label = tk.Label(window, text="Name", font=label_font, bg="lightblue")
t1_label.place(relx=0.2, rely=0.4, anchor='center')
t1 = tk.Entry(window, width=30, bd=5)
t1.place(relx=0.35, rely=0.4, anchor='center')

t2_label = tk.Label(window, text="Age", font=label_font, bg="lightblue")
t2_label.place(relx=0.2, rely=0.5, anchor='center')
t2 = tk.Entry(window, width=30, bd=5)
t2.place(relx=0.35, rely=0.5, anchor='center')

t3_label = tk.Label(window, text="Address", font=label_font, bg="lightblue")
t3_label.place(relx=0.2, rely=0.6, anchor='center')
t3 = tk.Entry(window, width=30, bd=5)
t3.place(relx=0.35, rely=0.6, anchor='center')

train_button = tk.Button(window, text="Train Classifier", font=label_font, bg='orange', fg='white', command=train_classifier, relief=tk.GROOVE)
train_button.place(relx=0.3, rely=0.8, anchor='center')

detect_button = tk.Button(window, text="Detect Face", font=label_font, bg='green', fg='white', command=detect_face, relief=tk.GROOVE)
detect_button.place(relx=0.5, rely=0.8, anchor='center')

generate_button = tk.Button(window, text="Generate Dataset", font=label_font, bg='pink', fg='black', command=generate_dataset, relief=tk.GROOVE)
generate_button.place(relx=0.7, rely=0.8, anchor='center')

window.mainloop()
