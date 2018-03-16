# import all the necessary packages, pretty sure you need to pip install numpy and PIL.
import os
import cv2
import numpy as np
from PIL import Image

recg = cv2.face.LBPHFaceRecognizer_create()
path = './dataSet'


def GetImageId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNP = np.array(faceImg)
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNP)
        print (ID)
        IDs.append(ID)
        cv2.imshow("training", faceNP)
        cv2.waitKey(10)

    return IDs, faces


Ids, faces = GetImageId(path)
recg.train(faces, np.array(Ids))
recg.save('./recg/trainingData.yml')  # This is basically the "Cascade Classifier XML" file.
cv2.destroyAllWindows()