import pygame
import pyaudio
import numpy as np
from gtts import gTTS

# IF "stereo mix" not available, write your available mic name in micInpt
micInpt = "stereo mix"

pygame.init()
pygame.mixer.init()

width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("AI Audio + Real-time Sound Visualization")

# -----------------------------
# AI TEXT-TO-SPEECH FUNCTION
# -----------------------------
def speak(prompt):
    tts = gTTS(prompt)
    tts.save("ai_output.mp3")
    pygame.mixer.music.load("ai_output.mp3")
    pygame.mixer.music.play()

# Play AI audio at start
speak("Hello. This is an AI generated sound test for your visualizer.")

# -----------------------------
# SET UP AUDIO INPUT (STEREO MIX)
# -----------------------------
p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if micInpt in info["name"].lower():
        mic_index = i
        break
else:
    raise ValueError("Specified mic not found")

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

# -----------------------------
# MAIN LOOP
# -----------------------------
running = True
while running:
    data = stream.read(CHUNK)
    samples = np.frombuffer(data, dtype=np.int16)

    screen.fill((0, 0, 0))

    # Convert audio samples to screen coordinates
    y = (samples + 32768) * height / 65536

    # Draw waveform
    for i in range(1, len(y)):
        pygame.draw.line(screen, (255, 255, 255),
                         (i - 1, y[i - 1]),
                         (i, y[i]), 1)

    pygame.display.flip()

    # Quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# -----------------------------
# CLEANUP
# -----------------------------
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
