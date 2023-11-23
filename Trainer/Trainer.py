import cv2 as cv
import numpy as np
import os

class Trainer():
    def __init__(self, path, name):
        self.cascadeClassifier = cv.CascadeClassifier('assets/lib/haarcascade_frontalface_alt.xml')
        self.faceRecognizer = cv.face.EigenFaceRecognizer_create()
        self.path = path
        self.name = name
        self.count = 1
        self.dataset = os.path.join(path, 'images/')
        self.cap = cv.VideoCapture(0)
        
        if not os.path.exists(self.dataset):
            os.makedirs(self.dataset)

    def captureDataset(self, limit: int, wh: int):
        while True:
            ret, frame = self.cap.read()
            frame = cv.flip(frame, 1)
            f = frame.copy()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            rostros = self.cascadeClassifier.detectMultiScale(gray, 2.3, 3)
            for(x, y, w, h) in rostros:
                f = cv.rectangle(f, (x,y), (x+w, y+h), (255, 255, 255), 2)
                face = frame[y:y+h, x:x+w]
                face = cv.resize(face,  (wh, wh), interpolation=cv.INTER_AREA)
                if face.any():
                    if not self.count > limit:
                        cv.imwrite(self.dataset + str(self.count) + '.jpg', face)
                        self.count=self.count+1

            return f

    def getCount(self):
        return self.count
    
    def endCapture(self):
        self.cap.release()

    def train(self):
        if not os.listdir(self.dataset) == []:
            labels = []
            facesData = []
            label = 0 
            for image in os.listdir(self.dataset):
                labels.append(label)
                facesData.append(cv.imread(os.path.join(self.dataset, image),0))
                label = label + 1
            try:
                self.faceRecognizer.train(facesData, np.array(labels))
                self.faceRecognizer.write(os.path.join(self.path, f"{self.name}.xml"))
            except:
                raise Exception('Ocurrio un error al entrenar el modelo')
        else:
            raise Exception('No hay imagenes en el dataset')
    
    def deleteImages(self):
        try:
            if not os.path.exists(self.dataset):
                raise Exception('No existe la carpeta de imágenes')
            
            elif not os.listdir(self.dataset):
                raise Exception('No hay imágenes en el dataset')
            
            else:
                os.rmdir(self.dataset)

        except Exception as e:
            raise Exception(f'Ocurrio un error al eliminar las imagenes: {e}')