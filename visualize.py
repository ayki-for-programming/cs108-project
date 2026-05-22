import pygame
import pyaudio
import numpy as np
import os
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch

# 1. GENERATE AI MUSIC FROM TEXT PROMPT
print("Loading MusicGen AI Model (this may take a minute on first run)...")
# Using the small model so it runs faster and uses less memory
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Get user input for the music style
prompt = input("\nEnter a music prompt (e.g., 'heavy metal guitar solo' or 'calm ambient synth'): ")
print(f"Generating music for: '{prompt}'... Please wait...")

# Process the prompt and generate audio tokens
inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt",
)

# Generate 8 seconds of audio (increase max_new_tokens for longer audio)
audio_values = model.generate(**inputs, max_new_tokens=512)

# Convert AI output to a standard audio array
sampling_rate = model.config.audio_encoder.sampling_rate
audio_data = audio_values[0, 0].cpu().numpy()

# Save the generated music to a WAV file
music_filename = "ai_generated_music.wav"
wavfile.write(music_filename, rate=sampling_rate, data=audio_data)
print("Music generated successfully! Starting visualizer...")


# 2. INITIALIZE PYGAME & AUDIO MIXER
pygame.init()
pygame.mixer.init()

width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(f"Visualizer - Playing: {prompt}")

# Load and play the AI generated music
pygame.mixer.music.load(music_filename)
pygame.mixer.music.play()


# 3. SET UP MICROPHONE AUDIO INPUT (FOR VISUALIZER)
p = pyaudio.PyAudio()

try:
    mic_index = p.get_default_input_device_info()["index"]
except IOError:
    raise ValueError("No input device (microphone) found on this system.")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=CHUNK)


# 4. MAIN VISUALIZER LOOP
running = True
clock = pygame.time.Clock()

while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)
    except IOError:
        continue

    screen.fill((20, 20, 30)) 

    # Normalize samples to fit window height
    y = (samples / 32768.0) * (height / 2) + (height / 2)
    x_steps = np.linspace(0, width, len(y))

    # Draw smooth waveform
    for i in range(1, len(y)):
        pygame.draw.line(screen, (255, 55, 100), 
                         (int(x_steps[i - 1]), int(y[i - 1])),
                         (int(x_steps[i]), int(y[i])), 2)

    pygame.display.flip()


# 5. CLEANUP
stream.stop_stream()
stream.close()
p.terminate()
pygame.mixer.music.stop()
pygame.quit()

# Delete the temporary music file
if os.path.exists(music_filename):
    os.remove(music_filename)
