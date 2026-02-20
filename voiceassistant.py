import pyaudio
from openwakeword.model import Model
import numpy as np
import time
from faster_whisper import WhisperModel
import vosk
import json
import os
import pygame
from piper import PiperVoice
from rapidfuzz import process, fuzz, utils
import datetime
from scipy.io import wavfile
import serial



#for Serial
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)


#reminder variables
water_flag = 0
last_water_time = time.time()
water_difference = 3600


#pygame mixer init
pygame.mixer.init(44100, -16, 2, 4096)  								
pygame.mixer.music.set_volume(0.50)


#ui tones
chime_tone = pygame.mixer.Sound("/home/pi/voice_assistant/sound_effects/chime.mp3")
menubutton_tone = pygame.mixer.Sound("/home/pi/voice_assistant/sound_effects/Menu Button.mp3")
menunotification_tone = pygame.mixer.Sound("/home/pi/voice_assistant/sound_effects/Menu Notification.mp3")


#music path
music_dir = "/home/pi/some_music"


#sound effects path
sound_dir = "/home/pi/voice_assistant/sound_effects"


#pyaudo  NOTE: speaking settings are not here, present in the next pyaudio block
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 3840
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

	
#cooldown to prevent multiple triggers for wake word
cooldown_time = 2
last_time = 0


#openwakeword settings
threshold = 0.3															#threshold for prediction
owwModel = Model(vad_threshold=0.2)


#some important settings, required for record_until_silence and sampling
silence_threshold = 0.1
max_silence = 1.5
output_rate = 16000
chunk_duration = CHUNK / RATE


#faster_whisper_settings
model_size = "tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8")


#vosk settings
target_rate = 16000
model = vosk.Model("/home/pi/voice_assistant/vosk-model-small-hi-0.22")


#piper settings
voice = PiperVoice.load("/home/pi/voice_assistant/hi_IN-pratham-medium.onnx")


#pyaudio speak
speak_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True, frames_per_buffer=1024)


#sound recorder settings
time_to_stop = 5
last_recorded = ""


#common recognized words
common_prefix = ("कृपया", "कृपा", "प्लीज")
common_extra = ("कर", "करके", "करो")
common_suffix = ("ना", "तो", "दो", "दें", "को")

#play music recognizer
id_music = ("म्यूजिक", "गाना", "दाना", "खाना", "गन्ना", "गहना", "कहना", "घाना")
trigger_music = ("चला", "चलाओ", "बजाओ", "बजा", "चलो", "जलाओ", "जला", "छला", "छल्ला", "चलना")

#stop music recognizer
id_stop = ("म्यूजिक", "गाना", "दाना", "खाना", "गन्ना", "गहना", "कहना")
trigger_stop = ("बंद")

#set volume recognizer
id_volume = ("वॉल्यूम", "आवाज़", "साउंड", "आवाज", "मात्रा")
trigger_volume = ("मीन", "नील", "शून्य", "दस", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे", "सौ", "मैक्स", "साथ", "फूल", "सो", "से", "सा")
extra_volume = ("एक")

#tell time recognizer:
id_time = ("वक्त", "समय", "टाइम", "बजा", "बजे", "बज", "बजी")
trigger_time = ("क्या", "कितने", "कितना", "बताओ", "बता", "कितनी", "किया", "गया")
extra_time = ("अभी", "हो", "है", "हैं", "रहा", "रहे", "बताओ", "बता", "हुआ", "हुई", "रही")

#tell day recognizer:
id_day = ("दिन", "वार")
trigger_day = ("कौन", "क्या", "कौनसी", "किया", "गया")
extra_day = ("आज", "है", "हैं", "हुआ", "हुई", "हो", "रहा", "रही", "सा", "सी", "हो", "अभी")

#tell date recognizer
id_date = ("तारीख", "तारिक")
trigger_date = ("कौन", "क्या", "कौनसी", "क्या", "कितने", "कितना", "बताओ", "बता", "कितनी", "किया", "गया")
extra_date = ("आज", "है", "हैं", "हुआ", "हुई", "हो", "रहा", "रही", "सा", "सी", "हो", "अभी")

#pause music
id_pause = ("म्यूजिक", "गाना", "दाना", "खाना", "गहना", "गन्ना", "कहना")
trigger_pause = ("रोको", "रोक", "रोग")

#unpause music
id_unpause = ("म्यूजिक", "गाना", "दाना", "खाना", "गन्ना", "गहना", "कहना")
trigger_unpause = ("शुरू")

#sound recording recognozer
id_record = ("रिकॉर्ड", "अभिलेख", "रेकॉर्डिंग", "रेकॉर्ड", "रिकॉर्डिंग")
trigger_record = ("करो", "ऑन", "शुरू", "कर")

#play last recording recognizer
id_playlastrecording = ("रिकॉर्ड", "अभिलेख", "रेकॉर्डिंग", "रेकॉर्ड", "रिकॉर्डिंग")
trigger_playlastrecording = ("चला", "चलाओ", "बजाओ", "बजा", "चलो", "जलाओ", "जला", "छला", "छल्ला", "चलना")
extra_playlastrecording = ("आखिरी", "पुरानी", "पुराना", "पिछला", "पिछली")

#water reminder
id_water = ("पानी", "जल")
trigger_water = ("याद")
extra_water = ("पान", "पीना", "मुझे", "दिलाना", "दिला", "दिलाओ", "की", "बात", "कि", "पीने", "खाने")
negation_water = ("मत", "नहीं", "ना")

#task schedule reader
id_task = ("काम", " कार्य")
trigger_task = ("कौन", "क्या", "कौनसी", "कितने", "कितना", "बताओ", "बता", "कितनी", "करना", "करने", "करनी", "किया", "गया")
trigger_tasktomorrow = ("कल")
extra_task = ("अभी", "आज", "हो", "है", "हैं", "रहा", "रहे", "रही", "सा", "सी", "से", "होगा", "होंगे", "होगी")

#light switch
id_light = ("लाइट", "बल्लभ", "बल्ब", "बत्ती")
trigger_light = ("ऑन", "शुरू", "जलाओ", "जला", "चलो", "चला", "चलाओ", "छला", "छल्ला", "चलना")
negation_light = ("ऑफ", "बंद", "बुझा", "भुजा")
extra_light = ("कर", "करो")


#temperature
id_temperature = ("तापमान", "टेम्परेचर", "मौसम", "गर्मी", "गर्म", "ठंडी", "ठंड", "ठंडा", "नमी", "ह्यूमिडिटी")
trigger_temperature = ("क्या", "कितने", "कितना", "बताओ", "बता", "कितनी", "किया", "कैसा", "कैसी", "किया", "गया")
extra_temperature = ("अभी", "आज", "हो", "है", "हैं", "रहा", "रहे", "रही", "सा", "सी", "से", "होगा", "होंगे", "होगी")


#emergency
id_emergency = ("इमरजेंसी", "हेल्प", "मदद", "हेल्थ", "हेल्प", "हेल्थ", "इमर्जेंसी")



def speak(text):
	for speak_chunk in voice.synthesize(text):
		speak_stream.write(speak_chunk.audio_int16_bytes)
		

def vosk_command_recg(audio_float32):
	audio_int16 = (audio_float32 * 32768).astype(np.int16)
	audio_bytes = audio_int16.tobytes()
	recognizer = vosk.KaldiRecognizer(model, target_rate)
	recognizer.AcceptWaveform(audio_bytes)
	result = json.loads(recognizer.Result())
	text = result.get("text", "")
	return text


def rms(audio_chunk):													#returns RMS
	return np.sqrt(np.mean(np.square(audio_chunk)))
	

def resample_numpy(audio_raw):											#resampling using numpy
	resampled = audio_raw.reshape(-1, 3).mean(axis=1).astype(np.int16)
	audio_resampled = resampled / 32768.0
	return audio_resampled
	

def record_until_silence():												#general purpose recording logic
	frames = []
	silence_time = 0.0
	print("recording")
	while True:
		data = mic_stream.read(CHUNK, exception_on_overflow=False)
		audio_raw = np.frombuffer(data, dtype=np.int16)
		audio_chunk = resample_numpy(audio_raw)
		frames.append(audio_chunk)
		level = rms(audio_chunk)
		if level < silence_threshold:
			silence_time += chunk_duration
		else:
			silence_time = 0.0
		
		if silence_time >= max_silence:
			print("silence detected")
			break		
	return np.concatenate(frames)
	
			
def transcribe(audio_chunk):											#for faster whisper
	print("transcribing...")
	model = WhisperModel(model_size, device="cpu", compute_type="int8")
	segments, info = model.transcribe(
		audio_chunk,
		language = "en",
		beam_size = 1
	)
	for segment in segments:
		print(segment.text)	
		return segment.text


def clean(raw_string):													#returns the raw string that is needed for further processing
	filtered_characters = []
	if raw_string is not None:
		for c in raw_string:
			if c.isalnum() or c==' ':
				filtered_characters.append(c)
	else:
		return ""
	processed = ''.join(filtered_characters)
	return processed.lstrip()


def play_music(song_input):
	received = song_input.lower()
	for file in os.listdir(music_dir):
		if file.lower().endswith((".mp3", ".wav", ".ogg")):
			if received in file.lower():
				print(">>ठीक है")
				if not pygame.mixer.music.get_busy():
					menunotification_tone.play()
				speak("ठीक है")
				file_path = os.path.join(music_dir, file)
				pygame.mixer.music.load(file_path)
				pygame.mixer.music.play(-1)
				print(f">>Now playing: {file}")
				return True
	return rapid_fuzz_search(song_input)
	
	
def rapid_fuzz_search(textin):
	print("searching...")
	songs = []
	supported = ('.mp3', '.wav', '.ogg')
	query = textin.strip()
	for filename in os.listdir(music_dir):
		lower_name = filename.lower()
		if (lower_name.endswith(supported)):
			songs.append(filename)
			
	search_names = []
	for s in songs:
		name_only = os.path.splitext(s)[0]
		search_names.append(name_only)
	search_result = process.extractOne(textin, search_names, processor=utils.default_process, scorer=fuzz.WRatio, score_cutoff=50)
	if search_result is not None:
		score = search_result[1]
		index = search_result[2]
		match = songs[index]
		print(f"Match found: {match}")
		print(f"Certaininty: {score:.1f}%")
		full_path = os.path.join(music_dir, match)
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.load(full_path)
		pygame.mixer.music.play(-1)
		print(f"Now playing: {songs[index]}")
		return True
	else:
		print(">>मुझे यह गाना नहीं मिला")
		speak("मुझे यह गाना नहीं मिला")
		return False


def stop_music():
	if not pygame.mixer.music.get_busy():
		print("गाना नहीं बज रहा है")			
		menunotification_tone.play()
		speak("गाना नहीं बज रहा है")
	else:
		print(">>ठीक है")
		speak("ठीक है")
		pygame.mixer.music.stop()
		print("Playback stopped  ◼")
	
	
def set_volume(target_volume):
	if target_volume in ("मीन", "नील", "शून्य"):
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.00)
		print("VOLUME:         0%")
	elif target_volume == "दस":
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()				
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.10)
		print("VOLUME: ▁       10%")
	elif target_volume == "बीस":
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.20)
		print("VOLUME: ▁       20%")
	elif target_volume == "तीस":
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.30)
		print("VOLUME: ▁▂      30%")
	elif target_volume == "चालीस":
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.40)
		print("VOLUME: ▁▂▃     40%")
	elif target_volume == "पचास":
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.50)
		print("VOLUME: ▁▂▃▄    50%")
	elif target_volume in ("साठ", "साथ"):
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.60)
		print("VOLUME: ▁▂▃▄    60%")
	elif target_volume in ("सत्तर"):
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.70)
		print("VOLUME: ▁▂▃▄▅   70%")
	elif target_volume == "अस्सी":
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.80)
		print("VOLUME: ▁▂▃▄▅▆  80%")
	elif target_volume == "नब्बे":
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(0.90)
		print("VOLUME: ▁▂▃▄▅▆  90%")
	elif target_volume in ("सौ", "मैक्स", "फूल", "सो", "से", "सा"):
		print(">>ठीक है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("ठीक है")
		pygame.mixer.music.set_volume(1.00)
		print("VOLUME: ▁▂▃▄▅▆▇ 100%")
	else:
		print("Volume Error")


def tell_time():
	dat = datetime.datetime.now()
	h24 = int(dat.strftime("%H"))
	h12 = int(dat.strftime("%I"))
	m = int(dat.strftime("%M"))
	if h24 >= 4 and h24 < 12:
		print(f">>अभी सुबह के {h12} बजकर {m} मिनट हो रहे हैं।")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"अभी सुबह के {h12} बजकर {m} मिनट हो रहे हैं।")
	if h24 >= 12 and h24 < 17:
		print(f">>अभी दोपहर के {h12} बजकर {m} मिनट हो रहे हैं।")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"अभी दोपहर के {h12} बजकर {m} मिनट हो रहे हैं।")
	elif h24 >= 17 and h24 < 21:
		print(f">>अभी शाम के {h12} बजकर {m} मिनट हो रहे हैं।")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"अभी शाम के {h12} बजकर {m} मिनट हो रहे हैं।")
	elif (h24 >= 21) or (h24 >= 0 and h24 < 4):
		print(f">>अभी रात के {h12} बजकर {m} मिनट हो रहे हैं।")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"अभी रात के {h12} बजकर {m} मिनट हो रहे हैं।")


def tell_day():
	dat = datetime.datetime.now()
	today = dat.strftime("%a")
	if today == "Mon":
		day_hi = "सोमवार"
		print(f"आज {day_hi} हैे")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"आज {day_hi} हैे")
	elif today == "Tue":
		day_hi = "मंगलवार"
		print(f"आज {day_hi} हैे")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"आज {day_hi} हैे")
	elif today == "Wed":
		day_hi = "बुधवार"
		print(f"आज {day_hi} है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"आज {day_hi} हैे")
	elif today == "Thu":
		day_hi = "गुरुवार"
		print(f"आज {day_hi} हैे")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"आज {day_hi} हैे")
	elif today == "Fri":
		day_hi = "शुक्रवार"
		print(f"आज {day_hi} हैे")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"आज {day_hi} हैे")
	elif today == "Sat":
		day_hi = "शनिवार"
		print(f"आज {day_hi} हैे")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"आज {day_hi} हैे")
	elif today == "Sun":
		day_hi = "रविवार"
		print(f"आज {day_hi} हैे")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak(f"आज {day_hi} हैे")
	else:
		print("Unknown error in tell_date")
	

def tell_date():
	dat = datetime.datetime.now()
	dd = dat.strftime("%d")
	mm = dat.strftime("%b")
	yyyy = dat.strftime("%Y")
	month_hi = ""
	if mm == "Jan":
		month_hi = "जनवरी"
	if mm == "Feb":
		month_hi = "फ़रवरी"
	if mm == "Mar":
		month_hi = "मार्च"
	if mm == "Apr":
		month_hi = "अप्रैल"
	if mm == "May":
		month_hi = "मई"
	if mm == "Jun":
		month_hi = "जून"
	if mm == "Jul":
		month_hi = "जुलाई"
	if mm == "Aug":
		month_hi = "अगस्त"
	if mm == "Sep":
		month_hi = "सितम्बर"
	if mm == "Oct":
		month_hi = "अक्टूबर"
	if mm == "Nov":
		month_hi = "नवंबर"
	if mm == "Dec":
		month_hi = "दिसंबर"
	print(f"आज {dd} {month_hi} {yyyy} है")
	if not pygame.mixer.music.get_busy():
		menunotification_tone.play()
	speak(f"आज {dd} {month_hi} {yyyy} है")
	

def sound_recorder():
	pygame.mixer.music.pause()
	print("Music playback paused  ▐▐")
	print(">>ठीक है")
	if not pygame.mixer.music.get_busy():
		menunotification_tone.play()
	speak("ठीक है")
	frames = []
	silence_time = 0.0
	chime_tone.play()
	time.sleep(1)
	start_time = time.time()
	print("Recording clip... Recoding will stop automatically after 5s of silence, or after 1 min")
	while time.time() - start_time <= 60:
		data = mic_stream.read(CHUNK, exception_on_overflow=False)
		audio_raw = np.frombuffer(data, dtype=np.int16)
		audio_chunk = resample_numpy(audio_raw)
		frames.append(audio_chunk)
		level = rms(audio_chunk)
		if level < silence_threshold:
			silence_time += chunk_duration
		else:
			silence_time = 0.0
		
		if silence_time >= time_to_stop:
			print(f"{time_to_stop} seconds of silence detected. Stopping...")
			break
	global last_recorded
	last_recorded = datetime.datetime.now()
	wavfile.write(f"/home/pi/voice_assistant/sound_recordings/{last_recorded}.wav", 16000, np.concatenate(frames))
	print("Stopped Recording")
	chime_tone.play()
	print(">>रिकॉर्डिंग बंद कर दी गई है")
	speak("रिकॉर्डिंग बंद कर दी गई है")
	pygame.mixer.music.unpause()
	print("Music playback resumed  ▶")
	

def play_last_recording():
	print(">>ठीक है")
	if not pygame.mixer.music.get_busy():
		menunotification_tone.play()
	speak("ठीक है")
	if last_recorded != "":
		pygame.mixer.music.load(f"/home/pi/voice_assistant/sound_recordings/{last_recorded}.wav")
		pygame.mixer.music.play()
		print(f">>Now playing: {last_recorded}.wav")
		volume_now = int((pygame.mixer.music.get_volume())*100)
		print(f"Volume: {volume_now}%")
	else:
		print("मुझे कोई हालिया रिकॉर्डिंग नहीं मिली।")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("मुझे कोई हालिया रिकॉर्डिंग नहीं मिली।")
	return True


def water_reminder():
	global water_flag
	water_flag = 1
	global last_water_time
	last_water_time = time.time()

def water_remind():
	global last_water_time
	global last_time
	if(water_flag == 1 and time.time()-last_water_time >= water_difference):
		last_time = time.time()
		initial_volume = pygame.mixer.music.get_volume()
		if (initial_volume>0.10):
			pygame.mixer.music.set_volume(0.10)
			chime_tone.play()
		print("\n>>कृपया पानी पियें\n")
		speak("कृपया पानी पियें")
		pygame.mixer.music.set_volume(initial_volume)
		last_water_time = time.time()
		last_time = time.time()											#required to prevent unintentional wake word detections


def tell_tasks():
	schedule_path = "/home/pi/voice_assistant/task_schedule/task_schedule.txt"
	tod = datetime.datetime.now()
	curr_day = tod.strftime("%d")
	curr_mon = tod.strftime("%m")
	curr_year = tod.strftime("%Y")
	tasks = []
	file = open(schedule_path, 'r')
	for line in file:
		line = line.strip()
		if line.startswith("(") and line.endswith(")"):
			content = line[1:-1]
			parts = content.split(",")
			if (len(parts) == 4):
				day = parts[0].strip()
				month = parts[1].strip()
				year = parts[2].strip()
				task_text = parts[3].strip(' \"')
				if day == curr_day and month == curr_mon and year == curr_year:
					tasks.append(task_text)
	file.close()
	if tasks:
		print(">>आज आपको ये काम करने हैं")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("आज आपको ये काम करने हैं")
		for task in tasks:
			initial_volume = pygame.mixer.music.get_volume()
			if (initial_volume>0.10):
				pygame.mixer.music.set_volume(0.10)
			print(task)
			speak(task)
			pygame.mixer.music.set_volume(initial_volume)
	else:
		initial_volume = pygame.mixer.music.get_volume()
		if (initial_volume>0.10):
				pygame.mixer.music.set_volume(0.10)	
		print(">>आपकी कार्य सूची में कोई कार्य नहीं है")
		speak("आपकी कार्य सूची में कोई कार्य नहीं है")
		pygame.mixer.music.set_volume(initial_volume)


def tell_tomorrowstasks():
	schedule_path = "/home/pi/voice_assistant/task_schedule/task_schedule.txt"
	tom = datetime.datetime.now() + datetime.timedelta(days=1)
	curr_day = tom.strftime("%d")
	curr_mon = tom.strftime("%m")
	curr_year = tom.strftime("%Y")
	tasks = []
	file = open(schedule_path, 'r')
	for line in file:
		line = line.strip()
		if line.startswith("(") and line.endswith(")"):
			content = line[1:-1]
			parts = content.split(",")
			if (len(parts) == 4):
				day = parts[0].strip()
				month = parts[1].strip()
				year = parts[2].strip()
				task_text = parts[3].strip(' \"')
				if day == curr_day and month == curr_mon and year == curr_year:
					tasks.append(task_text)
	file.close()
	if tasks:
		print(">>कल आपको ये काम करने हैं")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("कल आपको ये काम करने हैं")
		for task in tasks:
			print(task)
			speak(task)
	else:
		print(">>आपकी कार्य सूची में कोई कार्य नहीं है")
		if not pygame.mixer.music.get_busy():
			menunotification_tone.play()
		speak("आपकी कार्य सूची में कोई कार्य नहीं है")

def light_switch_on():
	ser.write(b's')
	print(">>मैंने लाइट जला दी है।")
	if not pygame.mixer.music.get_busy():
		chime_tone.play()
	speak("मैंने लाइट जला दी है।")

def light_switch_off():
	ser.write(b'r')
	print(">>मैंने लाइट बंद कर दी है।")
	if not pygame.mixer.music.get_busy():
		chime_tone.play()
	speak("मैंने लाइट बंद कर दी है।")
	
def sense():
	ser.write(b't')
	line = ser.readline().decode('utf-8').strip()
	
	if line:
		return line
	return False	


def call_help():
	ser.write(b'q')
	print(">>मदद आ रही है।")
	speak("मदद आ रही है")
	while True:
		chime_tone.play()
		time.sleep(2)


#greetings
print("\n"*5)
print("System is up and running\n")
print(">>नमस्ते")
speak("नमस्ते")
#begins here
while True:
	mic_audio = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
	resampled_audio = mic_audio.reshape(-1, 3).mean(axis=1).astype(np.int16)
	prediction = owwModel.predict(resampled_audio)
	water_remind()
	for mdl in prediction.keys():
		if(time.time()-last_time>cooldown_time and (mdl=="hey_mycroft" and prediction[mdl] >= threshold)):
			last_time = time.time()										#required to prevent unintentional wake word detections
			initial_volume = pygame.mixer.music.get_volume()
			if (initial_volume>0.10):
				pygame.mixer.music.set_volume(0.10)
			print(">>कहिये")
			if not pygame.mixer.music.get_busy():
				menubutton_tone.play()
			speak("कहिये")
			audio_data = record_until_silence()
			raw_text = vosk_command_recg(audio_data)
			print(raw_text)
			pygame.mixer.music.set_volume(initial_volume)
			id_music_counter = 0
			trigger_music_counter = 0
			id_time_counter = 0
			trigger_time_counter = 0
			id_day_counter = 0
			trigger_day_counter = 0
			id_stop_counter = 0
			trigger_stop_counter = 0
			id_volume_counter = 0
			trigger_volume_counter = 0
			id_pause_counter = 0
			trigger_pause_counter = 0
			id_unpause_counter = 0
			trigger_unpause_counter = 0
			id_date_counter = 0
			trigger_date_counter = 0
			id_record_counter = 0
			trigger_record_counter = 0
			id_playlastrecording_counter = 0
			trigger_playlastrecording_counter = 0
			id_water_counter = 0
			trigger_water_counter = 0
			negation_water_counter = 0
			id_task_counter = 0
			trigger_task_counter = 0
			trigger_tasktomorrow_counter = 0
			id_light_counter = 0
			trigger_light_counter = 0
			negation_light_counter = 0
			id_temperature_counter = 0
			trigger_temperature_counter = 0
			target_volume = ""
			for key in raw_text.split():
				if key in id_emergency:
					call_help()
			for key in raw_text.split():
				flag = 0
				if key in id_music:
					id_music_counter += 1	
					flag = 1
				if key in id_time:
					id_time_counter += 1
					flag = 1
				if key in id_day:
					id_day_counter += 1
					flag = 1	
				if key in id_stop:
					id_stop_counter += 1
					flag = 1
				if key in id_volume:
					id_volume_counter += 1
					flag = 1
				if key in id_pause:
					id_pause_counter += 1
					flag = 1
				if key in id_unpause:
					id_unpause_counter += 1
					flag = 1
				if key in id_date:
					id_date_counter += 1
					flag = 1
				if key in id_record:
					id_record_counter += 1
					flag = 1
				if key in id_playlastrecording:
					id_playlastrecording_counter += 1
					flag = 1
				if key in id_water:
					id_water_counter += 1
					flag = 1
				if key in id_task:
					id_task_counter += 1
					flag = 1
				if key in id_light:
					id_light_counter += 1
					flag = 1
				if key in id_temperature:
					id_temperature_counter += 1
					flag = 1
				if key in  trigger_music:
					trigger_music_counter += 1
					flag = 1	
				if key in trigger_time:
					trigger_time_counter += 1
					flag = 1	
				if key in trigger_day:
					trigger_day_counter += 1
					flag = 1	
				if key in trigger_stop:
					trigger_stop_counter += 1
					flag = 1
				if key in trigger_volume:
					target_volume = key
					trigger_volume_counter += 1
					flag = 1
				if key in trigger_pause:
					trigger_pause_counter += 1
					flag = 1
				if key in trigger_unpause:
					trigger_unpause_counter += 1
					flag = 1
				if key in trigger_date:
					trigger_date_counter += 1
					flag = 1
				if key in trigger_record:
					trigger_record_counter += 1
					flag = 1
				if key in trigger_playlastrecording:
					trigger_playlastrecording_counter += 1
					flag = 1
				if key in trigger_water:
					trigger_water_counter += 1
					flag = 1
				if key in negation_water:
					negation_water_counter += 1
					flag = 1
				if key in trigger_task:
					trigger_task_counter += 1
					flag = 1
				if key in trigger_tasktomorrow:
					trigger_tasktomorrow_counter += 1
					flag = 1
				if key in trigger_light:
					trigger_light_counter += 1
					flag = 1
				if key in negation_light:
					negation_light_counter += 1
					flag = 1
				if key in trigger_temperature:
					trigger_temperature_counter += 1
					flag = 1
				if key in extra_time:
					continue
				if key in extra_day:
					continue
				if key in extra_date:
					continue
				if key in extra_volume:
					continue
				if key in extra_playlastrecording:
					continue
				if key in extra_water:
					continue
				if key in extra_task:
					continue
				if key in extra_light:
					continue
				if key in extra_temperature:
					continue
				if key in common_prefix:
					continue
				if key in common_extra:
					continue
				if key in common_suffix:
					continue
				if key == " " or key == "":
					continue
				if flag == 0:
					id_music_counter = 0
					trigger_music_counter = 0
					id_time_counter = 0
					trigger_time_counter = 0
					id_day_counter = 0
					trigger_day_counter = 0
					id_stop_counter = 0
					trigger_stop_counter = 0
					#id_volume_counter = 0        						#keep this statement as a comment in order to receive volume instructions 
					trigger_volume_counter = 0
					id_pause_counter = 0
					trigger_pause_counter = 0
					id_unpause_counter = 0
					trigger_unpause_counter = 0
					id_date_counter = 0
					trigger_date_counter = 0
					id_record_counter = 0
					trigger_record_counter = 0
					id_playlastrecording_counter = 0
					trigger_playlastrecording_counter = 0
					id_water_counter = 0
					trigger_water_counter = 0
					negation_water_counter = 0
					id_task_counter = 0
					trigger_task_counter = 0
					trigger_tasktomorrow_counter = 0
					id_light_counter = 0
					trigger_light_counter = 0
					negation_light_counter = 0
					id_temperature_counter = 0
					trigger_temperature_counter = 0
					break												#this break is required to prevent continuing after foreign input
			if (id_music_counter>0 and trigger_music_counter>0):
				print(">>कौन सा?")
				initial_volume = pygame.mixer.music.get_volume()
				if (initial_volume>0.10):
					pygame.mixer.music.set_volume(0.10)
				if not pygame.mixer.music.get_busy():
					menunotification_tone.play()
				speak("कौन सा?")
				audio_data = record_until_silence()
				pygame.mixer.music.set_volume(initial_volume)
				raw_string = transcribe(audio_data)
				clean_string = clean(raw_string)
				if clean_string == "":
					print(">>>मैं आपकी बात समझ नहीं पाया। कृपया दोबारा प्रयास करे।")
					if not pygame.mixer.music.get_busy():
						menunotification_tone.play()
					speak("मैं आपकी बात समझ नहीं पाया। कृपया दोबारा प्रयास करे।")
					print("\n")
					last_time = time.time()
					break
				print(clean_string)
				checker = play_music(clean_string)
				if checker == True:
					print("▄ █ ▄ █ ▄ ▄ █ ▄ █ ▄ █")
					current_volume = int((pygame.mixer.music.get_volume())*100)
					print(f"Volume: {current_volume}%")
					
			elif (id_time_counter>0 and trigger_time_counter>0):
				tell_time()
			elif (id_day_counter>0 and trigger_day_counter>0):
				tell_day()
			elif (id_stop_counter>0 and trigger_stop_counter>0):
				stop_music()
			elif (id_volume_counter>0 and trigger_volume_counter>0):
				set_volume(target_volume)
			elif (id_volume_counter>0 and trigger_volume_counter==0):
				print("कृपया मात्रा को 10 के गुणकों में और 0 से 100 के बीच निर्दिष्ट करें।")
				if not pygame.mixer.music.get_busy():
					menunotification_tone.play()
				speak("कृपया मात्रा को 10 के गुणकों में और 0 से 100 के बीच निर्दिष्ट करें।")
			elif (id_pause_counter>0 and trigger_pause_counter>0):
				pygame.mixer.music.pause()
				print("Playback paused  ▐▐")
			elif (id_unpause_counter>0 and trigger_unpause_counter>0):
				pygame.mixer.music.unpause()
				print("Playback resumed  ▶")
			elif (id_date_counter>0 and trigger_date_counter>0):
				tell_date()
			elif (id_record_counter>0 and trigger_record_counter>0):
				sound_recorder()
			elif (id_playlastrecording_counter>0 and trigger_playlastrecording_counter>0):
				play_last_recording()
			elif (id_water_counter>0 and trigger_water_counter> 0 and negation_water_counter==0):
				water_flag = 1
				print(">>ठीक है, मैं आपको हर एक घंटे बाद पानी पीने की बात याद दिला दूंगा")
				if not pygame.mixer.music.get_busy():
					menunotification_tone.play()
				speak("ठीक है, मैं आपको हर एक घंटे बाद पानी पीने की बात याद दिला दूंगा")
				last_water_time = time.time()
			elif (id_water_counter>0 and trigger_water_counter> 0 and negation_water_counter>0):
				water_flag = 0
				print(">>ठीक है, मैं अब से आपको पानी पीने की बात याद नहीं दिलाऊंगा")
				if not pygame.mixer.music.get_busy():
					menunotification_tone.play()
				speak("ठीक है, मैं अब से आपको पानी पीने की बात याद नहीं दिलाऊंगा")
				last_water_time = time.time()
			elif (id_task_counter>0 and trigger_task_counter>0 and trigger_tasktomorrow_counter==0):
				tell_tasks()
			elif (id_task_counter>0 and trigger_task_counter>0 and trigger_tasktomorrow_counter>0):
				tell_tomorrowstasks()
			elif (id_light_counter>0 and trigger_light_counter>0 and negation_light_counter==0):
				light_switch_on()
			elif (id_light_counter>0 and negation_light_counter>0 and trigger_light_counter==0):
				light_switch_off()
			elif (id_temperature_counter>0 and trigger_temperature_counter>0):
				temp_data = sense()
				if temp_data:
					clean_line = temp_data.replace("(", "").replace(")", "")
					parts = clean_line.split(",")
					temp_float = float(parts[0])
					humidity_float = float(parts[1])
					initial_volume = pygame.mixer.music.get_volume()
					if (initial_volume>0.10):
						pygame.mixer.music.set_volume(0.10)
					print(f">>अभी तापमान {temp_float} डिग्री सेल्सियस एवं नमी {humidity_float} प्रतिषद है।")
					if not pygame.mixer.music.get_busy():
						menunotification_tone.play()
					speak(f"अभी तापमान {temp_float} डिग्री सेल्सियस एवं नमी {humidity_float} प्रतिषद है।")
					pygame.mixer.music.set_volume(initial_volume)
					
			else:
				print(">>मैं आपकी बात समझ नहीं पाया। कृपया दोबारा प्रयास करे।")
				if not pygame.mixer.music.get_busy():
					menunotification_tone.play()
				speak("मैं आपकी बात समझ नहीं पाया। कृपया दोबारा प्रयास करे।")
			
			print("...\n")
			
			
			last_time = time.time()										#removing may cause unintentional wake word detections


