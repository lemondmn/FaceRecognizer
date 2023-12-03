import PySimpleGUI as sg
import os
from recognizer import Recognizer

class RecognizerSetup:
    def __init__(self) -> None:
        self.file = None
        self.ip = None
        self.layout = [
            [sg.Text('Configuración')],
            [sg.Text('Archivo Eigenface\t'), sg.InputText(key='eigen', expand_x=True), sg.FileBrowse()],
            [sg.Text('Dirección IP\t'), sg.InputText(key='ip', expand_x=True)],
            [sg.Button('Cerrar'), sg.Button('Iniciar')],
        ]

    def run(self):
        window = sg.Window('Reconocimiento Facial', self.layout)
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == 'Cerrar':
                break
            elif event == 'Iniciar':
                self.file = values['eigen']
                self.ip = values['ip']
                if not os.path.exists(self.file):
                    sg.Popup('Error', 'El directorio no existe')                    
                else:
                    break
        window.close()
        return self.file, self.ip

class MainUI:
    def __init__(self, path, ip) -> None:
        self.path = path
        self.ip = ip
        self.recognizer = Recognizer(self.path, str(self.ip))
        
        # if not ip == None:
        #     if not self.recognizer.testConnection():
        #         sg.Popup('Error', 'No se pudo establecer conexión con la cámara IP')
        #         self.recognizer.destroy()
        #         self.recognizer = None
        #     else:
        #         sg.Popup('Éxito', 'Se estableció conexión con la cámara IP')

        self.layout = [
            [sg.Image(filename='', key='image')],
        ]
    
    def run(self):
        window = sg.Window('Reconocimiento Facial', self.layout)
        while True:
            event, values = window.read(timeout=20)
            if event == sg.WIN_CLOSED:
                break
            else:
                imgbytes = self.recognizer.getFrame()
                window['image'].update(data=imgbytes)
    
        self.recognizer.destroy()
        window.close()