# Face Recognizer

This project has two parts, the first one is a trainer and the second part is a predictor.

The trainer generates a `.xml` file that serves for the second part, the predictor, which takes one (or more) pre-trained `.xml` files and predices who's on the camera.

## Initializing

- Create and activate a Python Virtual enviroment.

```
python -m venv venv
venv\Scripts\activate
```

- Install the following dependencies inside the virtual enviroment:

OpenCV Contrib: `pip install opencv-contrib-python`
Eel: `pip install eel[jinja2]`
Pillow: `pip install Pillow`
