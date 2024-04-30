# FETCHING-UP-THE-TEXT-FROM-THE-VOICE-AS-A-CLOSED-CAPTION-USING-THE-HMM-MODEL-AND-NEURAL-NETWORKS
This project is a simple web application built with Streamlit for transcribing speech to text. It allows users to either record audio from their microphone or upload pre-recorded audio files for transcription. The application utilizes WhisperAI models for speech recognition.

# Features
Record audio from microphone and transcribe it to text.
Upload pre-recorded audio files (supported formats: WAV, MP3, M4A) for transcription.
Select from available WhisperAI models for transcription.
View transcription results in real-time.
Save transcription results to a text file.

# How to Use
Ensure you have Python installed.
Install the required dependencies using pip install -r requirements.txt.
Run the application by executing streamlit run app.py in your terminal.
The application will open in your default web browser.
Select a WhisperAI model from the dropdown menu.
Choose your input mode: either microphone or file upload.
Start recording or upload your audio file.
Click "Transcribe" to begin transcription.
View the transcription results on the screen.
Optionally, save the transcription to a text file.

# Dependencies
Python 3.x
Streamlit
Torch
WhisperAI
# Model Selection
It uses several pre-trained WhisperAI models for speech transcription. These models vary in size and performance, allowing users to choose based on their requirements.
