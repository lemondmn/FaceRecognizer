import cv2 as cv
import requests

class Recognizer:
    def __init__(self, path, ip) -> None:
        self.path = path
        self.ip = ip
        self.face = cv.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
        self.upperbody = cv.CascadeClassifier('haar/haarcascade_upperbody.xml')
        self.lowerbody = cv.CascadeClassifier('haar/haarcascade_lowerbody.xml')
        self.fullbody = cv.CascadeClassifier('haar/haarcascade_fullbody.xml')

        if self.ip == None:
            self.cap = cv.VideoCapture(0)
        else:
            self.cap = cv.VideoCapture(f'http://{self.ip}:81/stream')

        self.recognizer = cv.face.EigenFaceRecognizer_create()
        self.recognizer.read(str(self.path))
        
    def testConnection(self):
        tst = requests.get(f'http://{self.ip}:81/stream')

        if tst.status_code == 200:
            return True
        else:
            return False
    
    def capture(self):
        ret, frame = self.cap.read()
        if ret == False: raise Exception('No se pudo obtener el frame')
        frame = cv.flip(frame, 1)
        return frame
    
    def predict(self):
        frame = self.capture()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cpGray = gray.copy()
        rostros = self.face.detectMultiScale(gray, 1.3, 3)
        for(x, y, w, h) in rostros:
            frame2 = cpGray[y:y+h, x:x+w]
            frame2 = cv.resize(frame2,  (200,200), interpolation=cv.INTER_CUBIC)
            result = self.recognizer.predict(frame2)
            cv.putText(frame, '{}'.format(result), (x,y-20), 1,3.3, (255,255,0), 1, cv.LINE_AA)

        imgbytes = cv.imencode('.png', frame)[1].tobytes()
        return imgbytes
    
    def searchObjects(self):
        frame = self.capture()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, 1.3, 3)
        bodies = self.fullbody.detectMultiScale(frame, scaleFactor=1.5, minSize=(50,50))

        for(x, y, w, h) in faces:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.putText(frame, 'Rostro', (x,y-20), 1,3.3, (0,255,0), 1, cv.LINE_AA)
        for(x, y, w, h) in bodies:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            cv.putText(frame, 'Cuerpo', (x,y-20), 1,3.3, (0,0,255), 1, cv.LINE_AA)
        
        return frame
    
    def getFrame(self):
        frame = self.searchObjects()
        imgbytes = cv.imencode('.png', frame)[1].tobytes()
        return imgbytes
    
    def destroy(self):
        self.cap.release()
        cv.destroyAllWindows()