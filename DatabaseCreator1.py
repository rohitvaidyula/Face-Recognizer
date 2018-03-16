import os
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
id = input('enter user id: ')  # You can input "1" for a one person, "2" for the next and so on.
sample_size = 0

try:
    os.mkdir('./recg')
except OSError:
    print()

try:
    os.mkdir('./dataSet')
except OSError:
    print()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sample_size += 1
        cv2.imwrite("./dataSet/User." + str(id) + "." + str(sample_size) + ".jpg", gray[y:y + h, x:x + w])
        # Make sure that a file named dataSet is present in the location of this code.

        print("image ", sample_size)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.waitKey(10)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if sample_size > 100:  # You can change the number for a larger dataset.
        break

cap.release()
cv2.destroyAllWindows()