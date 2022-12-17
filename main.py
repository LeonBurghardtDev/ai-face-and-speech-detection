"""
Speech Detection

This script uses the OpenCV, Azure Speech Recognition, gTTS, and VLC libraries
to detect faces in the webcam video feed and recognize speech in the audio input.
It generates a response to the user's speech using OpenAI's GPT-3 API, and plays
the generated speech using the VLC library. It also displays a GUI with a textbox
and buttons to change the language for speech recognition.

Requirements:
- OpenCV
- Azure Speech Recognition
- gTTS
- VLC
- tkinter
- OpenAI API key

Usage:
1. Set the environment variable `azure_api_key` to your Azure API key.
2. Set the environment variable `openai_api_key` to your OpenAI API key.
3. Run the script: `python main.py`
4. The script will start the webcam and display the video feed in a window.
5. When a face is detected in the video, the script will create a rectangle around it.
6. When speech is detected, it will transcribe the speech to text, generate a response,
   generate speech from the response text, and play the generated speech.
7. The transcribed speech and generated response will be displayed in the GUI textbox.
8. Click the "English" or "German" button in the GUI to change the language for speech recognition.
9. Press "Q" in runtime to end the script.


Copyright @ Leon Burghardt, 2022. ALl rights reserved.

"""


import cv2
import os
import tkinter as tk
import tkinter.messagebox as messagebox
import time
import re
import sys
import threading
import openai
import vlc
import azure.cognitiveservices.speech as speechsdk
from gtts import gTTS

# Load the Haarcascade for face detection
cwd = os.getcwd()
script_path = os.path.realpath(__file__)
script_name = os.path.basename(script_path)
face_cascade = cv2.CascadeClassifier(script_path.replace(script_name,"")+'data/haarcascade_frontalface_default.xml')

# Set up the SpeechRecognitionClient object
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('azure_api_key'), region="eastus")
speech_config.speech_recognition_language="en-US"

audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

#open ai api key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Open the webcam
cap = cv2.VideoCapture(0)

# Open window for messages and setup the textbox aswell as the scrollbar
root = tk.Tk()

# dont start if camera is not available
if not cap.isOpened():
    messagebox.showerror("Error", "Cannot open camera")
    exit()

root.title("Speech Detection")
root.geometry("400x400")

info_label = tk.Label(root, text="Current language: "+speech_config.speech_recognition_language)
info_label.pack(side=tk.TOP)

copyright_label = tk.Label(root, text="Copyright © Leon Burghardt, 2022. All rights reserved.")
copyright_label.pack(side=tk.BOTTOM)

text_box = tk.Text(root)
scrollbar = tk.Scrollbar(root)
text_box.config(state="normal")

lang_button_english = tk.Button(root, text="English", command=lambda: change_language("en-US"))
lang_button_english.pack(side=tk.TOP)

lang_button_german = tk.Button(root, text="German", command=lambda: change_language("de-DE"))
lang_button_german.pack(side=tk.TOP)



# Configure the scrollbar to control the text box
scrollbar.config(command=text_box.yview)
text_box.config(yscrollcommand=scrollbar.set)


# Pack the widgets
text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)



text_box.pack(fill=tk.BOTH, expand=True)
text_box.insert("1.0", "Waiting for speech recognition...")


# Start the timer for fps calculation
start_time = time.time()
frame_counter = 0

# global variables
lang = "en"

def run_webcam():
    while True:
        # Read a frame from the webcam
        _, frame = cap.read()

        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:

            # Draw a rectangle around the face, depending on the size of the face the color of the rectangle changes
            if(w > 100 and h > 100):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0, 0), 2)
            elif(w < 150 and h < 150):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)	

            cv2.putText(frame, f'P: {x}/{y}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
            cv2.putText(frame, f'S: {w}/{h}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

        # Calculate the fps
        global frame_counter
        frame_counter += 1
        elapsed_time = time.time() - start_time
        fps = frame_counter / elapsed_time

        # Get the resolution of the video stream
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Draw the fps and resolution on the frame
        cv2.putText(frame, f'FPS: {fps:.1f} / Resolution: {width}x{height}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
        
        # Show the frame in a window
        cv2.imshow('Face Detection', frame)

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
            # Release the webcam
            cap.release()
            # Close all windows
            cv2.destroyAllWindows()
            root.destroy()
            sys.exit()
            
def run_speech_recognition():
    first_input = True;
    while True:
        # Start listening for speech
        speech_recognition_result = speech_recognizer.recognize_once_async().get()
        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Update the text box
                if first_input:
                    text_box.delete("1.0",tk.END)
                    text_box.insert("1.0", "Human: "+speech_recognition_result.text + "\n")
                    scroll_down()
                    first_input = False
                    answer_process(speech_recognition_result.text)

                else:
                    text_box.insert(tk.END, "\nHuman: "+speech_recognition_result.text + "\n")
                    scroll_down()
                    answer_process(speech_recognition_result.text)

# ai answer process
def answer_process(message):

    # Get the response from the AI and display it in the text box
    response = calc_response(message)
    text_box.insert(tk.END,"\nAI: "+str(response.replace(".",".\n"))+ "\n")

    # Scroll down to the last line
    scroll_down()

    # Convert the text to speech
    global lang
    tts = gTTS(response, lang=lang)

    # Create a temporary folder to save the audio file
    temp_dir = script_path.replace(script_name,"")+"/temp"

    # Create the temporary folder if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # save the audio file to the temporary folder
    path = (script_path.replace(script_name,"")+"temp/temp_"+".mp3").replace("\\","/")
    tts.save(path) 

    # Play the audio
    player = vlc.MediaPlayer(path)
    player.play()

# Change the language of the speech recognition and the text to speech
def change_language(language):
    speech_config.speech_recognition_language=language
    global speech_recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    global lang
    lang = language.split("-")[0]
    

# Calculate the response of the AI via OpenAI
def calc_response(message):
    # response object 
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=str(message),
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6
    )
    # format the response and return it
    regex = r"\n+"
    return re.sub(regex, "",response['choices'][0]['text'])

# Scroll down to the last line of the text box
def scroll_down():
    # Get the current position of the text box
    current_position = text_box.yview()[0]

    # Get the maximum possible position of the text box
    max_position = text_box.yview()[1]

    # If the current position is not at the bottom, scroll down
    if current_position < max_position:
        text_box.yview_moveto(max_position)


# Create and start the webcam and speech recognition threads
webcam_thread = threading.Thread(target=run_webcam)
speech_recognition_thread = threading.Thread(target=run_speech_recognition)
webcam_thread.start()
speech_recognition_thread.start()
root.mainloop()

# Wait for the threads to finish
webcam_thread.join()
speech_recognition_thread.join()


