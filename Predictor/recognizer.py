import cv2 as cv
import requests

class Recognizer:
    def __init__(self, path, ip) -> None:
        self.path = path
        self.ip = ip
        self.face = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

        if not self.ip == None:
            self.cap = cv.VideoCapture(f'http://{self.ip}:81/stream')
            print(f'http://{self.ip}:81/stream')
        else:
            self.cap = cv.VideoCapture(0)

        self.recognizer = cv.face.EigenFaceRecognizer_create()
        self.recognizer.read(str(self.path))
        
    def testConnection(self):
        try:
            requests.get(f'http://{self.ip}:81/stream')
            return True
        except:
            return False
    
    def getFrame(self):
        ret, frame = self.cap.read()
        frame = cv.flip(frame, 1)
        if ret == False: raise Exception('No se pudo obtener el frame')
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
    
    def destroy(self):
        self.cap.release()
        cv.destroyAllWindows()