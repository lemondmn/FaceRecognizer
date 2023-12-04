import PySimpleGUI as sg
import os
from recognizer import HaarRecognizer, YoloRecognizer

class RecognizerSetup:
    def __init__(self) -> None:
        self.layout = [
            [sg.Text('Configuración')],
            [sg.Text('Modo\t\t'), sg.Combo(['Haar', 'Yolo'], key='mode', default_value='Yolo', expand_x=True)],
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
                path = values['mode']
                ip = values['ip']
                if not os.path.exists(path):
                    sg.Popup('Error', 'El directorio no existe')                    
                else:
                    break
        window.close()
        return path, ip

class MainUI:
    def __init__(self, mode, ip) -> None:
        if ip == '': ip = None

        if mode == 'Haar':
            self.recognizer = HaarRecognizer(ip)
        elif mode == 'Yolo':
            self.recognizer = YoloRecognizer(ip)
        else:
            raise Exception('Modo no soportado')

        self.layout = [
            [sg.Image(filename='', key='image')],
            [sg.Button('Cerrar', expand_x=True)],
        ]
    
    def run(self):
        window = sg.Window('Reconocimiento Facial', self.layout)
        while True:
            event, values = window.read(timeout=20)
            if event == sg.WIN_CLOSED or event == 'Cerrar':
                break
            else:
                imgbytes = self.recognizer.getFrame()
                window['image'].update(data=imgbytes)
    
        self.recognizer.destroy()
        window.close()