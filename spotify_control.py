"""
@author: kirstaylo
"""

import os
import csv
import time
import random as r
import spotipy
import spotipy.util as util
import firebase_admin as fa
from firebase_admin import db


class SpotifyControl():
    def __init__(self, username, client_id, client_secret, redirect, device_id, genre_list):
        # assign global variables from init parameters
        self.username = username
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect = redirect
        self.device_id = device_id
        self.genre_list = genre_list
        # begin a count of the number of songs played within a session and current session id
        self.songs_played = 0
        self.session_id = None
        
        # instantuate the song dictionary containing song info for the current song 
        self.song_dict = {'timestamp_start' : None, 
                          'timestamp_end' : None, 
                          'name' : None, 
                          'artist' : None, 
                          'uri' : None, 
                          'genre': None,
                          'danceability' : None,
                          'energy' : None,
                          'key' : None,
                          'loudness' : None,
                          'mode' : None,
                          'speechiness' : None,
                          'acousticness' : None,
                          'instrumentalness' : None,
                          'liveness' : None,
                          'valence' : None,
                          'tempo' : None,
                          'time_signature' : None,
                          }


    ''' factory method to generate tokens for specific scopes '''
    def create_token(self, scope):
        token = util.prompt_for_user_token(self.username, scope, self.client_id, self.client_secret, self.redirect)
        sp = spotipy.Spotify(auth=token)
        return sp
    
    
    ''' method to generate specific tokens for the main scopes being used '''
    def gen_use_tokens(self):
        # create user read currently playing token
        self.urcp = self.create_token('user-read-currently-playing')
        # create user read playback state token
        self.urps = self.create_token('user-read-playback-state')
        # create user modify playback state token
        self.urms = self.create_token('user-modify-playback-state')


    ''' method to check that music is being played on the correct device at same volume '''
    def check_device(self):
        device_info = self.urps.devices()
        print(device_info)
        # iterate through devices to find the active one
        for device in device_info['devices']:
            if device['is_active']:
                # check that the music is playing from the correct device
                if device['id'] != self.device_id:
                    print('incorrect device connected')
                # check that the music is at full volume (or change it to full volume)
                if device['volume_percent'] != 100:
                    print('correcting volume')
                    self.urms.volume(100, device_id=self.device_id)
                 
                    
    ''' method to check the session id and instantiate csv for data collection '''
    def init_collection_session(self):
        # find which listening session this is
        self.session_id = len(os.listdir('./spotify_data')) + 1

        # open the file in the write mode
        with open('./spotify_data/song_data_session_{}.csv'.format(self.session_id), 'w', newline='') as f:
            # create the csv writer
            writer = csv.DictWriter(f, fieldnames=list(self.song_dict.keys()))
            writer.writeheader()


    ''' method to choose random song from spotify and to play it '''
    def play_random_song(self, min_song_duration, rick_roll_threshold):
        found_song = False
        search_count = 0
        
        # repeat loop until a valid song is found
        while not found_song:
            # create list of vowels
            vowels = ['a', 'e', 'i', 'o', 'u']
            # choose random vowel, genre and offset for search query
            vowel = r.choice(vowels)
            genre = r.choice(self.genre_list)
            o = r.randint(0,5)
            
            # search spotify with the randomised query
            results = self.urcp.search(q='track:{} genre:{}'.format(vowel, genre), type='track', limit=1, offset=o)
            try: 
                # extract the song data
                # this will throw an error if nothing is found
                rand_name = results['tracks']['items'][0]['name']
                rand_artist = results['tracks']['items'][0]['artists'][0]['name']
                rand_uri = results['tracks']['items'][0]['uri']
                
                # find the song duration
                duration = results['tracks']['items'][0]['duration_ms']
                # if the song is too short, throw an error
                if duration < min_song_duration:
                    raise IndexError
                    
                # print the name of the song being played and the song number
                print('Song {}: playing {} by {}'.format(self.songs_played+1, rand_name, rand_artist))
                # start playing the song from the middle
                self.urms.start_playback(uris=[rand_uri], position_ms=duration/2)
                
                # take note of time the song started playing
                timestamp = time.time()
                # increment the songs played count
                self.songs_played += 1
                # break out of the search loop
                found_song = True
               
            # if no song is found
            except:
                # try again with new search query
                search_count +=1
                # if it is taking more than the rick roll threshold to find a random song, get Rick Rolled
                if search_count > rick_roll_threshold:
                    print('playing Never Gonna Give You Up by Rick Astley')
                    self.urms.start_playback(uris=['spotify:track:4cOdK2wGLETKBW3PvgPWqT'])
                    
        # append known song info to the song dictionary
        self.song_dict['timestamp_start'] = timestamp
        self.song_dict['name'] = rand_name
        self.song_dict['artist'] = rand_artist
        self.song_dict['uri'] = rand_uri
        self.song_dict['genre'] = genre


    ''' method to extract audio data from song and combine into dict '''
    def extract_audio_data(self):
        # extract the song's audio features
        audio_features = self.urcp.audio_features(self.song_dict['uri'])
        
        # add the audio feature info to the song dictionary
        self.song_dict['danceability' ] = audio_features[0]['danceability']
        self.song_dict['energy'] = audio_features[0]['energy']
        self.song_dict['key'] = audio_features[0]['key']
        self.song_dict['loudness'] = audio_features[0]['loudness']
        self.song_dict['mode'] = audio_features[0]['mode']
        self.song_dict['speechiness'] = audio_features[0]['speechiness']
        self.song_dict['acousticness'] = audio_features[0]['acousticness']
        self.song_dict['instrumentalness'] = audio_features[0]['instrumentalness']
        self.song_dict['liveness'] = audio_features[0]['liveness']
        self.song_dict['valence'] = audio_features[0]['valence']
        self.song_dict['tempo'] = audio_features[0]['tempo']
        self.song_dict['time_signature'] = audio_features[0]['time_signature']
        
    
    ''' method to stop the Spotify playback and take the time that the music stopped playing '''
    def stop_song(self):
        # stop the song
        self.urms.pause_playback()
        # take note of time the song stops playing
        timestamp = time.time()
        # append known song info to the song dictionary
        self.song_dict['timestamp_end'] = timestamp
      
        
    ''' method to add the data of the current song to the csv '''  
    def append_to_csv(self):
        # open the file in the write mode
        with open('./spotify_data/song_data_session_{}.csv'.format(self.session_id), 'a', newline='') as f:
            # create the csv writer
            writer = csv.DictWriter(f, fieldnames=list(self.song_dict.keys()))
        
            # write the data fileds to the csv file
            try:
                writer.writerow(self.song_dict)
            # if a Unicode error arises 
            except UnicodeEncodeError:
                self.song_dict['name'] = 'Unknown'
                self.song_dict['artist'] = 'Unknown'
                writer.writerow(self.song_dict)
         
            
    ''' method to establish connection to firebase app as well as the session bucket '''
    def auth_firebase(self, firebase_url):
        # fetch the service account key JSON file contents
        cred = fa.credentials.Certificate("./key.json")
        
        # initialize the firebase app if it hasn't already been initialized
        try:
            fa.initialize_app(cred)
        except:
            pass
        
        # reference the databse
        db_ref = db.reference('/', url=firebase_url)
        # create the session bucket
        self.sesh_ref = db_ref.child('session-{}'.format(self.session_id))
    
    
    ''' method to post song data to firebase bucket '''
    def post_firebase(self):
        # create the song bucket
        song_ref = self.sesh_ref.child('song-{}'.format(self.songs_played))
        # push the song info
        song_ref.set(self.song_dict)

