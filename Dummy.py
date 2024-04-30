import os
import time
import io
from io import BytesIO
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import torch
import whisper
import torch.nn as nn
import torch.nn.functional as F

# Define the Hybrid HMM-NN model
class HybridHMMNN(nn.Module):
    def __init__(self):
        super(HybridHMMNN, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        # Define the HMM parameters
        self.num_states = 5
        self.num_outputs = 10
        self.transition_probs = torch.randn(self.num_states, self.num_states)
        self.emission_probs = torch.randn(self.num_states, self.num_outputs)

    def forward(self, x):
        # Implement the forward pass of the neural network
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Placeholder for the HMM decoding process
        # For demonstration purposes, we'll return random outputs
        output = torch.randint(0, self.num_outputs, (x.shape[0],))
        return output

# Function to load the Hybrid HMM-NN model
def load_hmm_nn_model(models_path):
    # Load Hybrid HMM-NN model
    hmm_nn_model = HybridHMMNN()  # Instantiate the Hybrid HMM-NN model
    hmm_nn_selected = True  # Set the model selection flag
    return hmm_nn_model, hmm_nn_selected

# Setup models storage path
models_path = st.secrets["MODELS_PATH"]

# Init vars
model_file = ''
whisper_file = '' 
audio_file = None

# Initialize Session State        
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
    
if 'whisper_selected' not in st.session_state:
    st.session_state.whisper_selected = False
    
if 'whisper_loaded' not in st.session_state:
    st.session_state.whisper_loaded = False
    
if 'model' not in st.session_state:
    st.session_state.model = None
    
if 'hmm_nn_model' not in st.session_state:
    st.session_state.hmm_nn_model = None
    
if 'hmm_nn_selected' not in st.session_state:
    st.session_state.hmm_nn_selected = False
    
# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    audio_data = None
    transcription = dict()
    
    # Streamlit UI: Title
    st.title("🗣 ⇢ TalkSee ⇢ 👀")
    # UI Columns
    col1, col2 = st.columns(2)
    
    # Select Model
    model_select = st.sidebar.selectbox("Select Model", ('WhisperAI', 'Hybrid HMM-NN'))

    if model_select == 'WhisperAI':
        with col1:
            st.header("WhisperAI Model")
            st.write("This is the WhisperAI model for speech transcription.")
        # Load WhisperAI model
        if st.session_state.model is None:
            st.session_state.model, st.session_state.whisper_selected = model_exists(models_path, col1, col2)
    
    elif model_select == 'Hybrid HMM-NN':
        with col1:
            st.header("Hybrid HMM-NN Model")
            st.write("This model combines a neural network with a hidden Markov model.")
        # Load Hybrid HMM-NN model
        if st.session_state.hmm_nn_model is None:
            st.session_state.hmm_nn_model, st.session_state.hmm_nn_selected = load_hmm_nn_model(models_path)
        
    # Select Input Mode
    with col2:
        st.header("Select Input Mode")
        input_type = st.radio(
            'Select Input Mode',
            ('Mic', 'File'),
            label_visibility='collapsed',
            horizontal=True
        ) 
            
        # Get User Input
        with col2:
            if input_type == 'Mic':
                #  Setup User Mic Input
                audio_data = setup_mic()
                if audio_data is None: 
                    st.write("Click 👆 to start mic recording...")
    
            else:
                #  Setup User File Input
                audio_data = setup_file(col2)
        
    # Setup UI
    transcription_placeholder = st.empty()
    
    with col1:
        if audio_data is not None and st.button('Transcribe', use_container_width=True):
            transcription = transcribe(audio_data, model_select)
            save_to_text_file(transcription["text"])
            # Render UI
            st.header("✍️ Transcription")
            transcription_placeholder.markdown(transcription["text"])

# Function to check if the WhisperAI model exists and load it
def model_exists(models_path, _col1, _col2):
    whisper_select = st.sidebar.selectbox(
        'Available Multilingual Models',
        ('tiny', 'base', 'small', 'medium', 'large', 'large-v2'),
        help="""
            |  Size  | Parameters | Multilingual model | Required VRAM | Relative speed |
            |:------:|:----------:|:------------------:|:-------------:|:--------------:|
            |  tiny  |    39 M    |       `tiny`       |     ~1 GB     |      ~32x      |
            |  base  |    74 M    |       `base`       |     ~1 GB     |      ~16x      |
            | small  |   244 M    |      `small`       |     ~2 GB     |      ~6x       |
            | medium |   769 M    |      `medium`      |     ~5 GB     |      ~2x       |
            | large  |   1550 M   |      `large`       |    ~10 GB     |       1x       |
        """,
        label_visibility='visible'
    )
    whisper_file = os.path.join(models_path, f"{whisper_select}.pt")
    whisper_selected = None
    if whisper_select != st.session_state.whisper_selected or st.session_state.whisper_loaded != True:
        model = whisper.load_model(whisper_select, device=DEVICE, download_root=models_path)
        if model:
            st.session_state.whisper_loaded = True
        return model, whisper_selected

# Function to setup mic input
def setup_mic():
    audio_bytes = audio_recorder(
        text='',
        recording_color="#a34bff",
        neutral_color="#000",
        icon_name="microphone-lines", 
        icon_size='7x',
        pause_threshold=2.0, 
        sample_rate=41_000
    )
    if audio_bytes:
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)
        recorded_file = BytesIO(audio_bytes)
        recorded_file.name = 'output.wav'
        recorded_file.type = 'audio/wav'
        recorded_file.id = len(recorded_file.getvalue()) if st.session_state.audio_file is not None else 0
        recorded_file.size = len(audio_bytes)
        st.session_state.audio_file = recorded_file
        
        if recorded_file:
            st.header("🎧 Recorded File")
            st.audio(st.session_state.audio_file)
        
    return st.session_state.audio_file if st.session_state.audio_file else None

# Function to setup file input
def setup_file(col2):
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Audio File", 
            key="uploaded_audio_file",
            type=["wav", "mp3", "m4a"],
            label_visibility='collapsed'
        )
        if uploaded_file:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state.audio_file = uploaded_file
            st.header("🎧 Uploaded File")
            st.audio(st.session_state.audio_file)
                
    return st.session_state.audio_file if st.session_state.audio_file else None

# Function to transcribe audio using the selected model
def transcribe(audio_file, model_select):
    transcription = {}
    if model_select == 'WhisperAI':
        transcription = st.session_state.model.transcribe(audio_file.name)
    elif model_select == 'Hybrid HMM-NN':
        # Perform transcriptions using the Hybrid HMM-NN model
        if audio_file is not None:
            audio_data = torch.randn(10)  # Placeholder audio data
            transcription_output = st.session_state.hmm_nn_model(audio_data)
            # For demonstration purposes, generate a random transcription
            transcription_text = " ".join([str(torch.randint(0, 10, (1,)).item()) for _ in range(10)])
            transcription = {"text": transcription_text}
    return transcription

# Function to save transcription to a text file
def save_to_text_file(transcription):
    file_name = "transcription.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(transcription)
    st.success(f"Transcription saved to {file_name}")

# Run the main function
if __name__ == "__main__":
    main()