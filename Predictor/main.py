from gui import RecognizerSetup, MainUI

if __name__ == '__main__':
    setup = RecognizerSetup()
    path, ip = setup.run()
    if not path == None:
        window = MainUI(path, ip)
        window.run()
    