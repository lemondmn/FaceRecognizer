import cv2 as cv
import numpy as np
import requests

class YoloRecognizer:
    def __init__(self, ip):
        if ip == '': ip = None
        self.url = ip
        if ip == None:
            self.cap = cv.VideoCapture(0)
        else:
            self.cap = cv.VideoCapture(f"http://{ip}:81/stream")
        self.wht = 320
        self.conf_threshold = 0.5
        self.nms_threshold = 0.3
        self.model_config = 'yolo/yolov3.cfg'
        self.model_weights = 'yolo/yolov3.weights'
        self.classes_file = 'yolo/coco.names'
        self.class_names = []
        
        with open(self.classes_file, 'rt') as f:
            self.class_names = f.read().rstrip('\n').split('\n')

        self.net = cv.dnn.readNetFromDarknet(self.model_config, self.model_weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    
    def handleAlert(self):
        print('Persona encontrada')
        
    def searchObject(self, outputs, im):
        hT, wT, cT = im.shape
        bbox = []
        class_ids = []
        confs = []
        found_person = False
        
        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))
        
        indices = cv.dnn.NMSBoxes(bbox, confs, self.conf_threshold, self.nms_threshold)
        
        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            if self.class_names[class_ids[i]] == 'person':
                found_person = True
                
            if self.class_names[class_ids[i]] == 'person':
                cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv.putText(im, f'{self.class_names[class_ids[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
        if found_person:
            self.handleAlert()
        
    def getFrame(self):
        ret, frame = self.cap.read()
        frame = cv.flip(frame, 1)

        blob = cv.dnn.blobFromImage(frame, 1/255, (self.wht, self.wht), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        outputs = self.net.forward(output_names)
        self.searchObject(outputs, frame)
        
        imgbytes = cv.imencode('.png', frame)[1].tobytes()
        return imgbytes
    
    def destroy(self):
        self.cap.release()
        cv.destroyAllWindows()

class HaarRecognizer:
    def __init__(self, ip) -> None:
        self.path = 'eigenface.xml'
        self.ip = ip
        self.face = cv.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
        self.upperbody = cv.CascadeClassifier('haar/haarcascade_upperbody.xml')
        self.lowerbody = cv.CascadeClassifier('haar/haarcascade_lowerbody.xml')
        self.fullbody = cv.CascadeClassifier('haar/haarcascade_fullbody.xml')

        if not self.ip == None:
            self.cap = cv.VideoCapture(f'http://{self.ip}:81/stream')
            print(f'http://{self.ip}:81/stream')
        else:
            self.cap = cv.VideoCapture(0)
    
    def getFrame(self):
        ret, frame = self.cap.read()
        frame = cv.flip(frame, 1)
        if ret == False: raise Exception('No se pudo obtener el frame')
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cpGray = gray.copy()
        rostros = self.face.detectMultiScale(gray, 1.3, 3)
        cuerpos = self.upperbody.detectMultiScale(gray, 1.3, 3)
        sups = self.upperbody.detectMultiScale(gray, 1.3, 3)
        infs = self.lowerbody.detectMultiScale(gray, 1.3, 3)

        for(x, y, w, h) in rostros:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        for (x, y, w, h) in cuerpos:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        for (x, y, w, h) in sups:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        for (x, y, w, h) in infs:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        imgbytes = cv.imencode('.png', frame)[1].tobytes()
        return imgbytes
    
    def destroy(self):
        self.cap.release()
        cv.destroyAllWindows()

class EigenRecognizer:
    def __init__(self, ip) -> None:
        self.path = 'eigenface.xml'
        self.ip = ip
        self.face = cv.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
        self.classnames = ['Carlos', 'Francisco', 'Humberto', 'Desconocido']

        if not self.ip == None:
            self.cap = cv.VideoCapture(f'http://{self.ip}:81/stream')
            print(f'http://{self.ip}:81/stream')
        else:
            self.cap = cv.VideoCapture(0)

        self.recognizer = cv.face.EigenFaceRecognizer_create()
        self.recognizer.read(self.path)
    
    def getFrame(self, treshold):
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
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            loss = round(result[1])

            if loss > treshold:
                cv.putText(frame, '{}'.format("Desconocido"), (x,y-20), 1,3.3, (0,0,255), 1, cv.LINE_AA)
            else:
                cv.putText(frame, '{}'.format(self.classnames[result[0]]), (x,y-20), 1,3.3, (0,255,0), 1, cv.LINE_AA)

        imgbytes = cv.imencode('.png', frame)[1].tobytes()
        return imgbytes
    
    def destroy(self):
        self.cap.release()
        cv.destroyAllWindows()

class EigenYoloRecognizer:
    def __init__(self, ip) -> None:
        if ip == '': ip = None
        self.url = ip
        if ip == None:
            self.cap = cv.VideoCapture(0)
        else:
            self.cap = cv.VideoCapture(f"http://{ip}:81/stream")

        # Yolo
        self.wht = 320
        self.conf_threshold = 0.5
        self.nms_threshold = 0.3
        self.model_config = 'yolo/yolov3.cfg'
        self.model_weights = 'yolo/yolov3.weights'
        self.classes_file = 'yolo/coco.names'
        self.class_names = []
        
        with open(self.classes_file, 'rt') as f:
            self.class_names = f.read().rstrip('\n').split('\n')

        self.net = cv.dnn.readNetFromDarknet(self.model_config, self.model_weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Eigenface
        self.path = 'eigenface.xml'
        self.ip = ip
        self.face = cv.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
        self.eigenclassnames = ['Carlos', 'Francisco', 'Humberto']

        self.recognizer = cv.face.EigenFaceRecognizer_create()
        self.recognizer.read(self.path)

    def searchPerson(self, outputs, im):
        hT, wT, cT = im.shape
        bbox = []
        class_ids = []
        confs = []
        found_person = False
        
        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))
        
        indices = cv.dnn.NMSBoxes(bbox, confs, self.conf_threshold, self.nms_threshold)
        
        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            if self.class_names[class_ids[i]] == 'person':
                found_person = True
                
            if self.class_names[class_ids[i]] == 'person':
                cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv.putText(im, f'{self.class_names[class_ids[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
    def searchFace(self, frame, treshold):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cpGray = gray.copy()
        rostros = self.face.detectMultiScale(gray, 1.3, 3)
        for(x, y, w, h) in rostros:
            frame2 = cpGray[y:y+h, x:x+w]
            frame2 = cv.resize(frame2,  (200,200), interpolation=cv.INTER_CUBIC)
            result = self.recognizer.predict(frame2)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            loss = round(result[1])

            if loss > treshold:
                cv.putText(frame, '{}'.format("Desconocido"), (x,y-20), 1,3.3, (0,0,255), 1, cv.LINE_AA)
            else:
                cv.putText(frame, '{}'.format(self.eigenclassnames[result[0]]), (x,y-20), 1,3.3, (0,255,0), 1, cv.LINE_AA)

    def getFrame(self, treshold):
        ret, frame = self.cap.read()
        frame = cv.flip(frame, 1)

        blob = cv.dnn.blobFromImage(frame, 1/255, (self.wht, self.wht), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        outputs = self.net.forward(output_names)
        self.searchPerson(outputs, frame)
        self.searchFace(frame, treshold)
        
        imgbytes = cv.imencode('.png', frame)[1].tobytes()
        return imgbytes

    def destroy(self):
        self.cap.release()
        cv.destroyAllWindows()