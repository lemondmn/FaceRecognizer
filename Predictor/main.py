from gui import RecognizerSetup, MainUI

if __name__ == '__main__':
    setup = RecognizerSetup()
    mode, ip = setup.run()
    window = MainUI(mode, ip)
    try:
        window.run()
    except Exception as e:
        print(e)
        window.recognizer.destroy()
        window.close()
