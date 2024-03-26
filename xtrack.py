
import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np
import csv

		
from IPython.display import Audio
from scipy.io import wavfile
from scipy import signal
from pathlib import Path
from pydub import AudioSegment


from utils import *

model = tf.saved_model.load('yamnet')
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

emotion_labels = ['Happy music',
 'Sad music',
 'Tender music',
 'Exciting music',
 'Angry music',
 'Scary music']
music_labels = ['Music', 
				'Musical instrument',
				'Singing',
				'Drum',
				'Rapping']
non_music_labels = ['Silence',
					'Speech',
				   'Narration, monologue',
					'Chatter',
				   'Cheering',
				   'Applause',
					]
neutral_labels = []
labels = music_labels+non_music_labels+neutral_labels

def processAudioFile(path):
	filename = str(path).split('/')[-1][:-4]
	print('processing', filename, '...')

	if 'wav' in str(path):
		sample_rate, data = wavfile.read(path, 'rb')
		original_sample_rate = sample_rate
		sample_rate, data = ensure_sample_rate(sample_rate, data)
	elif 'mp3' in str(path):
		# Read the MP3 file
		audio = AudioSegment.from_mp3(path)
		# Convert audio to 16kHz sampling rate
		sample_rate = 16000
		audio = audio.set_frame_rate(sample_rate)
		# Extract raw data and convert it to numpy array
		data = np.array(audio.get_array_of_samples())
	
	# Show some basic information about the audio.
	duration = len(data)/sample_rate
	print('Sample rate: %iHz'%sample_rate)
	print('Total duration:', time_format(duration))
	print('Size of the input: %i samples'%len(data))

	# # Listening to the wav file.
	# Audio(wav_data, rate=sample_rate)
	waveform = data / tf.int16.max
	# Run the model, check the output.
	scores, embeddings, spectrogram = model(waveform)
	# 51 scores per 48 samples (3ms)
	scores_np = scores.numpy()
	spectrogram_np = spectrogram.numpy()
	class_indices = []
	for l in labels:
		class_indices.append(class_names.index(l))
	probs = normalize_matrix(scores_np[:, class_indices])
	return filename, waveform, spectrogram_np, probs

def findMusicBorders(probs, dt = 0.48):
	music_starts = []
	music_stops = []
	
	consecutive_count = 0
	find_start = True
	find_stop = False
	
	patience = 0
	L_music = int(20/dt)
	L_nomusic = int(4/dt)
	L_patience = int(4/dt)
	music_count = 0
	nomusic_count = 0
	consecutive_count = 0
	music_indexes = [labels.index(l) for l in music_labels]
	non_music_indexes = [labels.index(l) for l in non_music_labels]
	for i, prob in enumerate(probs):
		music_prob = np.sum(prob[music_indexes])
		non_music_prob = np.sum(prob[non_music_indexes])
		ratio = music_prob/non_music_prob
		if find_start:
			if music_prob > 0.3:
				music_count += 1
				if music_count == L_music:
					music_starts.append((i-L_music-patience+1)*dt)
					find_start = False
					find_stop = True
					music_count = 0
			elif music_count>0:
				patience +=1
				if patience == L_patience:
					patience = 0 
					music_count = 0
			else:
				music_count = 0
		if find_stop:
			if music_prob < 0.1:
				nomusic_count += 1
				if nomusic_count == L_nomusic:
					music_stops.append((i-L_nomusic+1)*dt)
					find_stop = False
					find_start = True
					nomusic_count = 0
			else:
				nomusic_count = 0
	return music_starts, music_stops

def Xtrack(probs, dt=0.48, MUSIC_START_OFFSET = -0.1, MUSIC_STOP_OFFSET = 4.):
	T = len(probs)*dt
	music_starts, music_stops = findMusicBorders(probs)      
	all_onsets = sorted(music_starts+music_stops+[T])
	all_labels = []
	for i, o in enumerate(all_onsets[:-1]):
		next_onset = all_onsets[i+1]
		if o in music_starts:
			all_labels.append('music')
			if all_onsets[i]+MUSIC_START_OFFSET>0:
				all_onsets[i] += MUSIC_START_OFFSET
		
		if o in music_stops:
			all_labels.append('applause')
			all_onsets[i] = min(all_onsets[i]+MUSIC_STOP_OFFSET, next_onset-0.5)
			
			if len(np.where(np.sum(probs[0:15, [labels.index(x) for x in ['Speech', 'Narration, monologue']]], axis=1)>0.5)[0]) > 10/dt:
				all_labels[-1] = 'speech' 
				
			if np.mean(probs[int(o/dt):int(next_onset/dt), labels.index('Silence')])>0.5:
				all_labels[-1] = 'silence'

	return all_onsets[:-1], all_labels



def main(input_folder = 'Inputs/', output_path='Outputs/'):


	datapaths = []
	for p in Path(input_folder).rglob('*.wav'):
		datapaths.append(p)
	for p in Path(input_folder).rglob('*.mp3'):
		datapaths.append(p)


	for d in datapaths:
		

		filename, waveform, spectrogram_np, probs = processAudioFile(d)
		predicted_onsets, predicted_labels = Xtrack(probs)
		
		output_folder = output_path+filename+'/'
		create_directory(output_folder)
		audio_output_path = output_folder

		
		writeIndividualTracks(waveform, predicted_onsets, predicted_labels, output_path=output_folder)
		writeIndexesCSV(predicted_onsets, predicted_labels, output_path=output_folder)




if __name__ == "__main__":
	main()