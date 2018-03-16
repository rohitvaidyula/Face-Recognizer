# import the necessary packages
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()  # A special type of Cascade just for Faces
rec.read('./recg/trainingData.yml')  # The YML file is the actual trained algorithm
Id = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x,y,w,h) in faces:
        Id, conf = rec.predict(gray[y:y + h, x:x + w])  # Identifies if there's any recognizable  face from the dataset.
    if Id  == 1:
        Name = 'Rohit'
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, Name, (x, y + h), font, 2, (0, 255, 255))
    if Id != 1:
        print("Error, an unidentified face detected, sending out an Alert right now!!")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
# Name = str(Id)
# cv2.putText(frame, Name, (x, y + h), font, 2, (255, 0, 255))