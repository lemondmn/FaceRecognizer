import PySimpleGUI as sg
import recognizer as rec

class RecognizerSetup:
    def __init__(self) -> None:
        sg.theme('LightPurple')
        sg.set_options(font=('Helvetica', 14))

        self.combo_values = ['Algoritmo YOLOv3 (Personas)',
                             'Eigenface (Rostros)',
                             'Eigenface + YOLOv3 (Rostros + Personas)'
                             'Haar Cascade Classifier (Caras, cuerpo)'
                            ]
        self.layout = [
            [sg.Text('Configuración')],
            [sg.Text('Modo\t\t'), sg.Combo(self.combo_values, key='mode', expand_x=True, default_value=self.combo_values[0])],
            [sg.Text('Dirección IP\t'), sg.InputText(key='ip', expand_x=True)],
            [sg.Button('Cerrar', expand_x=True), sg.Button('Iniciar', expand_x=True)],
        ]

    def run(self):
        window = sg.Window('Reconocimiento Facial', self.layout)
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == 'Cerrar':
                break
            elif event == 'Iniciar':
                mode = self.combo_values.index(values['mode'])
                ip = values['ip']
                break
        window.close()
        return mode, ip

class MainUI:
    def __init__(self, mode, ip) -> None:
        sg.theme('LightPurple')
        sg.set_options(font=('Helvetica', 14))
        self.slider = [[sg.Text('Sensibilidad')],[sg.Slider(range=(5000, 15000), default_value=9000, resolution=500, orientation='h', key='threshold', expand_x=True)]]
        self.title = None

        if ip == '': ip = None

        self.layout = [
            [sg.Image(filename='', key='image')],
            [sg.Button('Cerrar', expand_x=True)],
        ]

        if mode == 0:
            self.recognizer = rec.YoloRecognizer(ip)
            self.title = 'YOLOv3 (Personas)'
        elif mode == 1:
            self.recognizer = rec.EigenRecognizer(ip)
            self.title = 'Eigenface (Rostros)'
            self.layout.append(self.slider)
        elif mode == 2:
            self.recognizer = rec.EigenYoloRecognizer(ip)
            self.title = 'Eigenface + YOLOv3 (Rostros + Personas)'
            self.layout.append(self.slider)
        elif mode == 3:
            self.recognizer = rec.HaarRecognizer(ip)
            self.title = 'Haar Cascade Classifier (Caras, cuerpo)'
        else:
            raise Exception('Modo no soportado')
        
        self.mode = mode
    
    def run(self):
        window = sg.Window(self.title, self.layout)
        while True:
            event, values = window.read(timeout=20)
            if event == sg.WIN_CLOSED or event == 'Cerrar':
                break
            else:
                if self.mode == 2 or self.mode == 3:
                    imgbytes = self.recognizer.getFrame(values['threshold'])
                else:
                    imgbytes = self.recognizer.getFrame()
                window['image'].update(data=imgbytes)
    
        self.recognizer.destroy()
        window.close()