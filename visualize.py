import pygame
import pyaudio
import numpy as np
import os
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch

# DEVICE SETUP

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# LOAD MUSIC MODEL

print("\nLoading MusicGen AI Model...")

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small"
).to(device)

# PROMPT

prompt = input("\nEnter a music prompt:\n\n")

print(f"\nGenerating music for: {prompt}")
print("Please wait...\n")

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt",
).to(device)

audio_values = model.generate(
    **inputs,
    max_new_tokens=256
)

sampling_rate = model.config.audio_encoder.sampling_rate
audio_data = audio_values[0, 0].cpu().numpy()

music_filename = "ai_generated_music.wav"

wavfile.write(music_filename, sampling_rate, audio_data)

print("Music generated! Starting visualizer...")

# PYGAME SETUP

pygame.init()
pygame.mixer.init()

width, height = 1000, 600

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Wave Visualizer")

clock = pygame.time.Clock()

pygame.mixer.music.load(music_filename)
pygame.mixer.music.play()

# AUDIO INPUT (MIC / SYSTEM INPUT DEVICE)

p = pyaudio.PyAudio()

CHUNK = 1024
RATE = 44100

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# SMOOTHING MEMORY

prev_y = np.zeros(CHUNK)

# STAR BACKGROUND

stars = np.random.randint(0, [width, height], (120, 2))

# MAIN LOOP

running = True

while running:

    clock.tick(45)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # READ AUDIO

    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)
    except:
        continue

    # BACKGROUND

    screen.fill((15, 10, 20))

    for star in stars:
        pygame.draw.circle(screen, (255, 255, 255), star, 1)

    # CLEAN WAVEFORM (FIXED CORE LOGIC)

    y = (samples / 32768.0) * (height / 2) + (height / 2)
    x_steps = np.linspace(0, width, len(y))

    # light smoothing only (prevents shaking)
    y = prev_y * 0.6 + y * 0.4
    prev_y = y

    # DRAW WAVEFORM (RED NEON)

    for i in range(1, len(y)):

        pygame.draw.line(
            screen,
            (255, 55, 100),
            (int(x_steps[i - 1]), int(y[i - 1])),
            (int(x_steps[i]), int(y[i])),
            2
        )

    pygame.display.flip()

# CLEANUP

stream.stop_stream()
stream.close()
p.terminate()

pygame.mixer.music.stop()
pygame.quit()

if os.path.exists(music_filename):
    os.remove(music_filename)