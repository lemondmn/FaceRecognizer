from gui import RecognizerSetup, MainUI

if __name__ == '__main__':
    setup = RecognizerSetup()
    mode, ip = setup.run()
    print(f"mode: {mode}, ip: {ip}")
    window = MainUI(mode, ip)
    try:
        window.run()
    except Exception as e:
        print(e)
