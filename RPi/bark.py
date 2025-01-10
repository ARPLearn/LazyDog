#!/usr/bin/env python3
import pygame
import time

def play_bark():
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Use absolute path
    sound_file = '/home/pi/WAVEGO/RPi/sounds/bark.mp3'
    
    try:
        print(f"Attempting to play: {sound_file}")
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        
        # Wait for the sound to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error playing sound: {e}")
    
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    play_bark()
