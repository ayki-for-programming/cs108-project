import pygame
import pyaudio
import numpy as np
from gtts import gTTS
import os

pygame.init()
pygame.mixer.init()

width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("AI Audio + Real-time Sound Visualization")

# 1. AI TEXT-TO-SPEECH FUNCTION
def speak(prompt):
    tts = gTTS(prompt)
    tts.save("ai_output.mp3")
    pygame.mixer.music.load("ai_output.mp3")
    pygame.mixer.music.play()

# Play AI audio at start
speak("Hello. This is an AI generated sound test for your visualizer.")

# 2. SET UP AUDIO INPUT (MAC COMPATIBLE)
p = pyaudio.PyAudio()

# Automatically fall back to the default input device on Mac
try:
    mic_index = p.get_default_input_device_info()["index"]
except IOError:
    raise ValueError("No input device (microphone) found on this system.")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# exception_on_overflow=False prevents crashes if the Mac drops audio frames
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=CHUNK)

# 3. MAIN LOOP
running = True
clock = pygame.time.Clock()

while running:
    # Limit frame rate to keep CPU usage low
    clock.tick(60)

    # Event handling first to keep window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    try:
        # exception_on_overflow prevents crashing on slower rendering loops
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)
    except IOError:
        continue

    screen.fill((20, 20, 30)) # Sleek dark blue background

    # Normalize samples to fit perfectly inside the window height
    # PyAudio int16 ranges from -32768 to 32767
    y = (samples / 32768.0) * (height / 2) + (height / 2)

    # Map chunk index to horizontal screen pixels
    x_steps = np.linspace(0, width, len(y))

    # Draw smooth waveform
    for i in range(1, len(y)):
        pygame.draw.line(screen, (0, 255, 200), # Neon cyan color
                         (int(x_steps[i - 1]), int(y[i - 1])),
                         (int(x_steps[i]), int(y[i])), 2)

    pygame.display.flip()

# 4. CLEANUP
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()

# Clean up the audio file after closing
if os.path.exists("ai_output.mp3"):
    os.remove("ai_output.mp3")
