import PySimpleGUI as sg
import os
from Trainer import Trainer

sg.theme('DarkBlue14')
sg.set_options(font=('Helvetica', 16))

layout = [
            [sg.Text('Entrenador de modelos de reconocimiento facial', expand_x = True)]
            ,[
                sg.Text('Estado: '),
                sg.InputText(key='status', expand_x = True, readonly=True)
            ]
            ,[
                sg.Text('Carpeta de trabajo\t\t'),
                sg.InputText(key='folder', expand_x = True),
                sg.FolderBrowse()
            ]
            ,[sg.HorizontalSeparator()]
            ,[sg.Text('Agregar nueva persona al dataset')]
            ,[
                sg.Text('Nombre de la persona\t'),
                sg.InputText(key='name',expand_x = True)
            ]
            ,[sg.Button('Iniciar captura', key='capture', expand_x = True)]
            ,[sg.HorizontalSeparator()]
            ,[sg.Text('Entrenamiento del modelo')]
            ,[sg.Button('Administrar datasets', key='admin', expand_x = True)]
            ,[
                sg.Button('Comenzar entrenamiento', key='train', expand_x = True),
                sg.Button('Abrir carpeta', key='openfolder', expand_x = True)
            ]
        ]
capture = [
            [sg.Text('', key='capt_title', expand_x = True)]
            ,[sg.Image(key='image')]
            ,[
                [
                    sg.Text('', expand_x=True),
                    sg.Text('', key='count', expand_x=True),
                    sg.Text('', expand_x=True)
                ]
            ]
        ]

window = sg.Window('Entrenador de modelos de reconocimiento facial', layout, location = (200,200))

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == 'capture':
        path = values['folder']
        name = values['name']
        fp = os.path.join(path, name)

        if path == '' or name == '':
            sg.popup('Favor de llenar todos los campos.')
        elif (os.path.exists(fp)):
            sg.popup('Ya existe un modelo o carpeta con ese nombre.')
        elif (os.path.exists(path)):
            pp = sg.Window('Captura de imagenes', capture, location = (200,200))
            
        
