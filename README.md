# Speech Detection

This script uses the OpenCV, Azure Speech Recognition, gTTS, and VLC libraries to detect faces in the webcam video feed and recognize speech in the audio input. It generates a response to the user's speech using OpenAI's GPT-3 API, and plays the generated speech using the VLC library. It also displays a GUI with a textbox and buttons to change the language for speech recognition.

## Requirements

- OpenCV
```python
pip install opencv-python
```
- Azure Speech Recognition
```python
pip install azure-cognitiveservices-speech
```
- gTTS
```python
pip install gTTS
```
- VLC
```download https://www.videolan.org/vlc/download-windows.en-GB.html
```
- tkinter
```python
pip install tk
```
- OpenAI API key
```python
pip install openai
```

## Usage


1. Set the environment variable `azure_api_key` to your Azure API key.
2. Set the environment variable `openai_api_key` to your OpenAI API key.
3. Run the script: `python main.py`
4. The script will start the webcam and display the video feed in a window.
5. When a face is detected in the video, the script will create a rectangle around it.
6. When speech is detected, it will transcribe the speech to text, generate a response, generate speech from the response text, and play the generated speech.
7. The transcribed speech and generated response will be displayed in the GUI textbox.
8. Click the "English" or "German" button in the GUI to change the language for speech recognition.
9. Press "Q" in runtime to end the script.

## Copyright

Copyright @ Leon Burghardt, 2022. All rights reserved.
