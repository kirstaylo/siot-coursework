"""
@author: kirst
"""

import os
import json
import time
import random as r
import pandas as pd
import spotipy
import spotipy.util as util
from dotenv import dotenv_values

# load in secret environmental variables from .env file
config = dotenv_values(".env")

# extract the environmental variables
CLIENT_ID = config['CLIENT_ID']
CLIENT_SECRET = config['CLIENT_SECRET']
USERNAME = config['USERNAME']
DEVICE_ID = config['DEVICE_ID']

REDIRECT = "http://localhost:8888/callback/"
PHASE_LEN = 15

''' function to generate tokens for specific scopes '''
def create_token(scope):
    token = util.prompt_for_user_token(USERNAME, scope, CLIENT_ID, CLIENT_SECRET, REDIRECT)
    sp = spotipy.Spotify(auth=token)
    return sp

# create user read currently playing token
sp_urcp = create_token('user-read-currently-playing')
# create user read playback state token
sp_urps = create_token('user-read-playback-state')
# create user modify playback state token
sp_urms = create_token('user-modify-playback-state')

''' function to check that music is being played on the correct headphones at same volume '''
def check_headphones():
    device_info = sp_urps.devices()
    print(device_info)
    # iterate through devices to find the active one
    for device in device_info['devices']:
        if device['is_active']:
            # check that the music is playing from the correct device
            if device['id'] != DEVICE_ID:
                print('incorrect device connected')
            # check that the music is at full volume (or change it to full volume)
            if device['volume_percent'] != 100:
                print('correcting volume')
                sp_urms.volume(100, device_id=DEVICE_ID)

''' function to choose random song from spotify and to play it '''
def play_random_song(genre_list):
    found_song = False
    search_count = 0
    
    # repeat loop until a valid song is found
    while not found_song:
        # create list of vowels
        vowels = ['a', 'e', 'i', 'o', 'u']
        # choose random vowel, genre and offset for search query
        vowel = r.choice(vowels)
        genre = r.choice(genre_list)
        o = r.randint(0,5)
        
        # search spotify with the randomised query
        results = sp_urcp.search(q='track:{} genre:{}'.format(vowel, genre), type='track', limit=1, offset=o)
        try: 
            # extract the song data
            # this will throw an error if nothing is found
            rand_name = results['tracks']['items'][0]['name']
            rand_artist = results['tracks']['items'][0]['artists'][0]['name']
            rand_uri = results['tracks']['items'][0]['uri']
            
            # find the song duration
            duration = results['tracks']['items'][0]['duration_ms']
            # if the song is too short (< 30 seconds), throw an error
            if duration < 60000:
                raise IndexError
            # start playing the song from the middle
            print('playing {} by {}'.format(rand_name, rand_artist))
            sp_urms.start_playback(uris=[rand_uri], position_ms=duration/2)
            timestamp = time.time()
            found_song = True
           
        # if no song is found
        except IndexError:
            # try again with new search query
            search_count +=1
            # if it is taking more than 100 tries to find a random song, get Rick Rolled
            if search_count > 100:
                print('playing Never Gonna Give You Up by Rick Astley')
                sp_urms.start_playback(uris=['spotify:track:4cOdK2wGLETKBW3PvgPWqT'])
    return rand_name, rand_artist, rand_uri, timestamp, genre

''' function to extract audio data from song and combine into dict '''
def current_song_info(name, artist, uri, timestamp, genre):
    # extract the song's audio features
    audio_features = sp_urcp.audio_features(uri)
    
    # create the song info dictionary
    song_dict = {'timestamp' : timestamp, 
                 'name' : name, 
                 'artist' : artist, 
                 'uri' : uri, 
                 'genre': genre,
                 'danceability' : audio_features[0]['danceability'],
                 'energy' : audio_features[0]['energy'],
                 'key' : audio_features[0]['key'],
                 'loudness' : audio_features[0]['loudness'],
                 'mode' : audio_features[0]['mode'],
                 'speechiness' : audio_features[0]['speechiness'],
                 'acousticness' : audio_features[0]['acousticness'],
                 'instrumentalness' : audio_features[0]['instrumentalness'],
                 'liveness' : audio_features[0]['liveness'],
                 'valence' : audio_features[0]['valence'],
                 'tempo' : audio_features[0]['tempo'],
                 'time_signature' : audio_features[0]['time_signature'],
                 }
    
    return song_dict

''' function to append new song info to data dictionary '''
def add_song_data(data, song):
    for key in data.keys():
        data[key].append(song[key])
    return data

''' function to export song data to csv file '''
def export_song_csv(data):
    # build a pandas dataframe from the data
    df = pd.DataFrame(data, columns = ['timestamp', 'name', 'artist', 'uri', 'genre', 'danceability', 'energy',
                                       'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
                                       'liveness', 'valence', 'tempo', 'time_signature'])
    
    # find the session number by counting the other files in the folder
    session = len(os.listdir('./spotify_data')) + 1
    # export the data to a csv file 
    df.to_csv('./spotify_data/song_data_session_{}.csv'.format(session), index=False) 
    print('csv exported')
    
    
''' main function for testing and data collection '''
def run_spotify_session():
    # extract the genre information from the genres json
    with open('spotify_genres.json', 'r') as genre_file:
        genre_list = json.load(genre_file)
    
    # check for correct device and volume
    check_headphones()
    
    # create empty data dictionary
    data = {
            'timestamp' : [], 
            'name' : [], 
            'artist' : [], 
            'uri' : [], 
            'genre': [],
            'danceability' : [],
            'energy' : [],
            'key' : [],
            'loudness' : [],
            'mode' : [],
            'speechiness' : [],
            'acousticness' : [],
            'instrumentalness' : [],
            'liveness' : [],
            'valence' : [],
            'tempo' : [],
            'time_signature' : []
            }
    
    # itz loopin' time
    try:
        while True:
           name, artist, uri, timestamp, genre = play_random_song(genre_list)
           song_dict = current_song_info(name, artist, uri, timestamp, genre)
           data = add_song_data(data, song_dict)
           time.sleep(PHASE_LEN)
    except KeyboardInterrupt:
        export_song_csv(data)
        
# run the session        
if __name__ == "__main__":
    run_spotify_session()


