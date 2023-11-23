import cv2 as cv
import numpy as np
import os

class Trainer():
    def __init__(self, path, name):
        self.cascadeClassifier = cv.CascadeClassifier('Assets/haarcascade_frontalface_default.xml')
        self.faceRecognizer = cv.face.EigenFaceRecognizer_create()
        self.path = path
        self.name = name
        self.dataset = path + 'images/'
        self.cap = cv.VideoCapture(0)
        
        if not os.path.exists(self.dataset):
            os.makedirs(self.dataset)

    def captureDataset(self, limit: int, wh: int):
        i = 1
        while True:
            ret, frame = self.cap.read()
            frame = cv.flip(frame, 1)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            rostros = self.cascadeClassifier.detectMultiScale(gray, 2.3, 3)
            for(x, y, w, h) in rostros:
                frame = cv.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 2)
                face = frame[y:y+h, x:x+w]
                face = cv.resize(face,  (wh, wh), interpolation=cv.INTER_AREA)
                if face.any():
                    if not i > limit:
                        cv.imwrite(self.dataset + str(i) + '.jpg', face)
                        i=i+1

            return frame
    
    def endCapture(self):
        self.cap.release()

    def train(self):
        if not os.listdir(self.dataset) == []:
            labels = []
            facesData = []
            label = 0 
            for image in os.listdir(self.dataset):
                labels.append(label)
                facesData.append(cv.imread(self.dataset+image,0))
                label = label + 1
            try:
                faceRecognizer = cv.face.EigenFaceRecognizer_create()
                faceRecognizer.train(facesData, np.array(labels))
                faceRecognizer.write(self.path + self.name + '.xml')
            except:
                raise Exception('Ocurrio un error al entrenar el modelo')
        else:
            raise Exception('No hay imagenes en el dataset')
    
    def deleteAllImages(self):
        try:
            if os.listdir(self.dataset) == []:
                raise Exception('No hay imagenes en el dataset')
            elif not os.exists(self.dataset):
                raise Exception('No existe la carpeta de imagenes')
            else:
                for image in os.listdir(self.dataset):
                    os.remove(self.dataset + image)
                os.remove(self.path)
        except:
            raise Exception('Ocurrio un error al eliminar las imagenes')