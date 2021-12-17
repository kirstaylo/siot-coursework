"""
@author: kirstaylo
"""

import json
import time
from dotenv import dotenv_values
from spotify_control import SpotifyControl

# load in secret environmental variables from .env file
config = dotenv_values(".env")

# extract the environmental variables to instantiate the class with
CLIENT_ID = config['CLIENT_ID']
CLIENT_SECRET = config['CLIENT_SECRET']
USERNAME = config['USERNAME']
DEVICE_ID = config['DEVICE_ID']
FB_URL = config['FB_URL']

REDIRECT = "http://localhost:3000/callback/"
PHASE_LEN = 15
MIN_SONG_DURATION_MS = 6000
RICK_ROLL_THRESHOLD = 100


# extract the genre information from the genres json
with open('spotify_genres.json', 'r') as genre_file:
    GENRE_LIST = json.load(genre_file)

# create the instance of spotify control class
SpotifySIOT = SpotifyControl(USERNAME, CLIENT_ID, CLIENT_SECRET, REDIRECT, DEVICE_ID, GENRE_LIST)


''' main function for testing and data collection '''
def run_spotify_session():
    
    # create authentication tokens for the different use scopes
    SpotifySIOT.gen_use_tokens()
    
    # check for correct device and volume
    SpotifySIOT.check_device()
    
    # check for appropriate listening session number and create a csv for the information collection
    SpotifySIOT.init_collection_session()
    
    # authenticate firebase access
    SpotifySIOT.auth_firebase(FB_URL)
    
    # loopin' time
    try:
        while True:
            # let the program time out to let the subject's EDA recover to find a baseline
            time.sleep(PHASE_LEN)
            
            # play a random song
            SpotifySIOT.play_random_song(MIN_SONG_DURATION_MS, RICK_ROLL_THRESHOLD)
            
            # extract the audio features from the song
            SpotifySIOT.extract_audio_data()

            # let the program time out and let the song play so EDA can react
            time.sleep(PHASE_LEN)
            
            # stop the music and record time of stopping music
            SpotifySIOT.stop_song()
            
            # post the song data to firebase
            SpotifySIOT.post_firebase()
            
    except KeyboardInterrupt:
        print('songs played: {}'.format(SpotifySIOT.songs_played))
        
        
# run the session        
if __name__ == "__main__":
    run_spotify_session()

