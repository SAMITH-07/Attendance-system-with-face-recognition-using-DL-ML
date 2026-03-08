from collections.abc import Iterable #Used for checking if something is an iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
from datetime import datetime


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
#Checks if the item is an Iterable and that it is not a string.
            for x in flatten(item):
# the function calls itself recursively (flatten(item)) to further explore any nested items
                yield x
        else:
            yield item


embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5

print("[INFO] loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

Roll_Number = ""
box = []
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

# Attendance tracking variables
attendance_logged = set()  # To avoid duplicate entries
attendance_file = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

# Create attendance file with headers if it doesn't exist
try:
    with open(attendance_file, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Roll_Number', 'Timestamp', 'Date'])
    print(f"[INFO] Created attendance file: {attendance_file}")
except FileExistsError:
    print(f"[INFO] Using existing attendance file: {attendance_file}")

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    box = np.append(box, row)
                    print("Box",box)
                    name = str(name)
                    if name in row:
                        person = str(row)
                        print(name)
                listString = str(box)
##                print(box)
                if name in listString:
                    singleList = list(flatten(box))
                    listlen = len(singleList)
                    print("listlen",listlen)
                    Index = singleList.index(name)
                    print("Index",Index)
                    name = singleList[Index]
                    Roll_Number = singleList[Index + 1]
                    print(Roll_Number)
                    
                    # Log attendance if not already logged for this person today
                    attendance_key = f"{name}_{Roll_Number}_{datetime.now().strftime('%Y-%m-%d')}"
                    if attendance_key not in attendance_logged and proba > 0.7:  # Higher confidence threshold for logging
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        date = datetime.now().strftime('%Y-%m-%d')
                        
                        # Write to attendance file
                        with open(attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([name, Roll_Number, timestamp, date])
                        
                        attendance_logged.add(attendance_key)
                        print(f"[ATTENDANCE] {name} ({Roll_Number}) marked at {timestamp}")
                        
#Displaying Results:
            text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            # Change color based on attendance status
            color = (0, 255, 0) if f"{name}_{Roll_Number}_{datetime.now().strftime('%Y-%m-%d')}" in attendance_logged else (0, 0, 255)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
