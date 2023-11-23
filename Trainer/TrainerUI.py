import tkinter as tk
from Trainer import Trainer

class TrainerUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('Nuevo modelo')
        self.master.geometry('500x500')
        self.master.resizable(False, False)
        self.create_widgets()
        self.pack()

    def create_widgets(self):
        self.lblName = tk.Label(self.master, text='Nombre del modelo')
        self.lblName.pack()
        self.txtName = tk.Entry(self.master)
        self.txtName.pack()
        self.lblLimit = tk.Label(self.master, text='Limite de captura')
        self.lblLimit.pack()
        self.txtLimit = tk.Entry(self.master)
        self.txtLimit.pack()
        self.lblWh = tk.Label(self.master, text='Tamaño de imagen (cuadrada, n*n)')
        self.lblWh.pack()
        self.txtWh = tk.Entry(self.master)
        self.txtWh.pack()
        self.btnStart = tk.Button(self.master, text='Iniciar', command=self.start)
        self.btnStart.pack()
        self.btnEnd = tk.Button(self.master, text='Terminar', command=self.end)
        self.btnEnd.pack()
        self.lblStatus = tk.Label(self.master, text='Status')
        self.lblStatus.pack()
        self.lblStatusValue = tk.Label(self.master, text='Esperando')
        self.lblStatusValue.pack()

    def start(self):
        try:
            self.trainer = Trainer('test/', self.txtName.get())
            self.trainer.captureDataset(int(self.txtLimit.get()), int(self.txtWh.get()))
            self.lblStatusValue['text'] = 'Capturando'
        except Exception as e:
            self.lblStatusValue['text'] = e

    def end(self):
        try:
            self.trainer.endCapture()
            self.trainer.train()
            self.lblStatusValue['text'] = 'Terminado'
        except Exception as e:
            self.lblStatusValue['text'] = e

if __name__ == '__main__':
    root = tk.Tk()
    app = TrainerUI(master=root)
    app.mainloop()