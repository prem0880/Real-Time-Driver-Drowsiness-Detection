#from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
from PIL import Image as im
import cv2
import math
import socket
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras
from PIL import Image, ImageOps

HOST = '192.168.172.84' 
PORT = 65431       

model = tensorflow.keras.models.load_model('keras_model.h5', compile=False)

def dista(a,b):
	x1 = a[0]
	y1 = a[1]
	x2 = b[0]
	y2 = b[1]
	return math.sqrt((x2-x1)**2+(y2-y1)**2)

def increase_brightness(img, value=30):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	lim = 255 - value
	v[v > lim] = 255
	v[v <= lim] += value
	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img
	
def check_luminance(frame):
	ycbcr = image.convert('YCbCr')
	ycbcr = np.ndarray((ycbcr.size[1], ycbcr.size[0], 3), 'u1', ycbcr.tobytes())
	y, cb, cr = cv2.split(ycbcr)
	sum1 = np.sum(y)
	total = np.size(y)
	#print(sum1)
	#print(total)
	M = sum1/total
	#print('average luminance :')
	#print(M)
	y_threshold = 60
	if M<y_threshold:
		return 1
	return 0

def histogram_equalization(img_in):
	b,g,r = cv2.split(img_in)
	equ_b = cv2.equalizeHist(b)
	equ_g = cv2.equalizeHist(g)
	equ_r = cv2.equalizeHist(r)
	equ = cv2.merge((equ_b, equ_g, equ_r))
	return equ

def sound_alarm(path):
	playsound.playsound(path)
	
def eye_aspect_ratio(eye):
	#A = dist.euclidean(eye[1], eye[5])
	A = dista(eye[1],eye[5])
	#B = dist.euclidean(eye[2], eye[4])
	B = dista(eye[2],eye[4])
	#C = dist.euclidean(eye[0], eye[3])
	C = dista(eye[0],eye[3])
	ear = (2.0 * C) / (A + B)
	return ear

def mouth_aspect_ratio(mouth):
	#A = dist.euclidean(mouth[2], mouth[10])
	A = dista(mouth[2], mouth[10])
	#B = dist.euclidean(mouth[4], mouth[8])
	B = dista(mouth[4], mouth[8])
	#C = dist.euclidean(mouth[0], mouth[6])
	C = dista(mouth[0], mouth[6])
	ear = (2.0 * C) / (A + B)
	return ear

def sleep_log():
	from firebase import firebase
	firebase = firebase.FirebaseApplication('https://pybase-11d1e-default-rtdb.firebaseio.com/', None)
	t = time.asctime( time.localtime(time.time()) )
	sleep_time = ''
	sleep_time = t
	data = {'id' : 107,
	'Name': 'xyz',
	'sleeptime':sleep_time
	}
	result = firebase.post('/pybase-11d1e-default-rtdb/sleeptime',data)
	print(result)
	
def model1(frame):
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(image)
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	size = (224, 224)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)
	image_array = np.asarray(image)
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
	data[0] = normalized_image_array
	predictions = model.predict(data)
	#print(predictions)
	score = predictions[0]
	res = 'open'
	if score[0]>score[1]:
		print('Going to sleep')
		res = 'Going to sleep'
	else:
		print('Awake')
		res = 'Awake'
	cv2.putText(frame, res, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
	cv2.putText(frame, "Awake Probability: {:.2f}".format(score[1]), (285, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
	cv2.putText(frame, "Sleep Probability: {:.2f}".format(score[0]), (295, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
	cv2.imshow("Detection", frame)
	
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True)
ap.add_argument("-a", "--alarm", type=str, default="")
ap.add_argument("-w", "--webcam", type=int, default=0)
args = vars(ap.parse_args())


EYE_AR_THRESH = 4
MOUTH_AR_THRESH = 2
EYE_AR_CONSEC_FRAMES = 6
MOUTH_AR_CONSEC_FRAMES = 6
COUNTER = 0
MCOUNTER = 0
ALARM_ON = False
POSITION = 4
DEVIATIONS = 0
OUT=1
IN=1
#0 - top
#1 - bottom
#2 - left
#3 - right

detector = dlib.get_frontal_face_detector()
#detector = dlib.simple_object_detector("detector1.svm")
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
mStart = 48
mEnd = 61

vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
cnt = 0

while True:
	cnt+=1
	frame = vs.read()
	if cnt%10==0:
		cnt=0
		model1(frame)
	frame = imutils.resize(frame, width=450)
	image = im.fromarray(frame)
	light = 0
	light = check_luminance(image)
	if light == 1:
		frame = histogram_equalization(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	det = 0
	for rect in rects: #0 - top,left ; 337 - right, 450 - bottom
		det = 1
		l = rect.left()
		t = rect.top()
		r = rect.right()
		b = rect.bottom()
		x = (l+r)/2
		y = (t+b)/2
		ec = str(x)+", "+str(y)
		#cv2.putText(frame, ec, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		if x<225: # left
			if y<168.5:
				if x<y: #left
					POSITION = 2
					cv2.putText(frame, "Left", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				else: #top
					POSITION = 0
					cv2.putText(frame, "Top", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				if x<337-y: #left
					POSITION = 2
					cv2.putText(frame, "Left", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				else: #bottom
					POSITION = 1
					cv2.putText(frame, "Bottom", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else: # right
			if y<168.5:
				if 450-x<y: #right
					POSITION = 3
					cv2.putText(frame, "Right", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				else: #top
					POSITION = 0
					cv2.putText(frame, "Top", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				if 450-x<337-y: #right
					POSITION = 3
					cv2.putText(frame, "Right", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				else: #bottom
					POSITION = 1
					cv2.putText(frame, "Bottom", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mar = mouth_aspect_ratio(mouth)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		if ear > EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				cv2.putText(frame, "DROWSINESS ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#sleep_log()
				#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
					#s.connect((HOST, PORT))
					#s.sendall(b'1')
					#data = s.recv(1024)
		else:
			COUNTER = 0
			ALARM_ON = False
		if mar < MOUTH_AR_THRESH:
			MCOUNTER += 1
			if MCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON = True
				cv2.putText(frame, "GETTING TIRED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				#	s.connect((HOST, PORT))
				#	s.sendall(b'2')
					#data = s.recv(1024)
		else:
			MCOUNTER = 0
			ALARM_ON = False
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	if det == 0:
		DEVIATIONS+=1
		OUT+=1
		if DEVIATIONS>2:
			if POSITION==0:
				cv2.putText(frame, "Tilt camera above", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				#	s.connect((HOST, PORT))
				#	s.sendall(b'3')
					#data = s.recv(1024)
			elif POSITION==2:
				cv2.putText(frame, "Tilt camera to the left", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				#	s.connect((HOST, PORT))
				#	s.sendall(b'5')
					#data = s.recv(1024)
			elif POSITION==1:
				cv2.putText(frame, "Tilt camera bottom", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				#	s.connect((HOST, PORT))
				#	s.sendall(b'4')
					#data = s.recv(1024)
			elif POSITION==3:
				cv2.putText(frame, "Tilt camera right", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
					#s.connect((HOST, PORT))
					#s.sendall(b'6')
					#data = s.recv(1024)
			else:
				cv2.putText(frame, "Get inside frame!", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)            
	else:
		DEVIATIONS = 0
		IN+=1
	for k, d in enumerate(rects):
		#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		cv2.rectangle(frame, (int(d.left()),int(d.top())), (int(d.right()),int(d.bottom())), (0,255,0), 2)
	cv2.putText(frame, "Deviation ratio: {:.2f}".format(OUT/IN), (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()
