import cv2 as cv
import numpy as np

class ObjectDetector:
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
        print('Alerta')
        
    def find_person(self, outputs, im):
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

        blob = cv.dnn.blobFromImage(frame, 1/255, (self.wht, self.wht), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        outputs = self.net.forward(output_names)
        self.find_person(outputs, frame)
        
        imgbytes = cv.imencode('.png', frame)[1].tobytes()
        return imgbytes

if __name__ == '__main__':
    detector = ObjectDetector('')
    while True:
        detector.getFrame()