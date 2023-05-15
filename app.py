import streamlit as st 
import cv2
import altair as alt
from PIL import Image,ImageEnhance
import numpy as np 
import pandas as pd 
import os 

import datetime

from keras.models import load_model,model_from_json 

import time


from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import win32com.client
import pythoncom 
import tempfile 


speaker = win32com.client.Dispatch("SAPI.SpVoice",pythoncom.CoInitialize())   




#importing the cnn model+using the CascadeClassifier to use features at once to check if a window is not a face region
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier("facerecog/haarcascade_frontal_face_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX 
FRAME_WINDOW = st.image([])
stframe = st.empty()  






def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)
	
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

			fc = gray[y:y+h, x:x+w]
			roi = cv2.resize(fc, (48, 48))
			pred = classifier.predict(roi[np.newaxis, :, :, np.newaxis])
			
			maxindex = int(np.argmax(pred))
			finalout = emotion_labels[maxindex]
			
			output = str(finalout)
			
			cv2.putText(img, output, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,cv2.LINE_AA)
			# cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img,faces,pred

	 




	
	












emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise'] 


classifier =load_model('model/model_78.h5')

# load weights into new model
classifier.load_weights("model/model_weights_78.h5")

#

class VideoTransformer(VideoTransformerBase):
	def transform(self, frame):
		img = frame.to_ndarray(format="bgr24")

		#image gray
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
			image=img_gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces:
			cv2.rectangle(img=img, pt1=(x, y), pt2=(
				x + w, y + h), color=(255, 0, 0), thickness=2)
			roi_gray = img_gray[y:y + h, x:x + w]
			roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
			if np.sum([roi_gray]) != 0:
				roi = roi_gray.astype('float') / 255.0
				roi = img_to_array(roi)
				roi = np.expand_dims(roi, axis=0)
				prediction = classifier.predict(roi)[0]
				maxindex = int(np.argmax(prediction))
				finalout = emotion_labels[maxindex]
				output = str(finalout)
			label_position = (x, y-10)
			cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		return img   



	



		

	







	


def main():
	
	

	activities = ["Images" ,"Videos","LiveWebcam","About"]
	choice = st.sidebar.selectbox("Select Activity",activities)

	

	
		
		
	
	
	if choice == 'Images':
		html_temp = """
	
			
		<marquee behavior="scroll" direction="left" width="100%;">
		<h2 style= "color: red; font-family: 'Raleway',sans-serif; font-size: 62px; font-weight: 800; line-height: 72px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Face Emotion Recognition System</h2>
		</marquee><br>
		"""
		st.markdown(html_temp, unsafe_allow_html=True)
		
		
		st.title("Face Expression WEB Application :")
		st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
		
	#if image if uploaded,display the progress bar +the image
		if image_file is not None:
				our_image = Image.open(image_file)
				st.text("Original Image")
				progress = st.progress(0)
				for i in range(100):
					time.sleep(0.01)
					progress.progress(i+1)
				st.image(our_image)
		if image_file is None:
			st.error("No image uploaded yet")

		# Face Detection
		task = ["Faces"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		if st.button("Process"):
			if feature_choice == 'Faces':

				#process bar
				progress = st.progress(0)
				for i in range(100):
					time.sleep(0.05)
					progress.progress(i+1)
				#end of process bar
				
				result_img,result_faces,prediction = detect_faces(our_image)
 
				
				# st.subheader(prediction)
				if st.image(result_img) :
					st.success("Found {} faces".format(len(result_faces))) 

		


			

				



	elif choice=="Videos":
		html_temp = """
	
			
		<marquee behavior="scroll" direction="left" width="100%;">
		<h2 style= "color: red; font-family: 'Raleway',sans-serif; font-size: 62px; font-weight: 800; line-height: 72px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Face Emotion Recognition System</h2>
		</marquee><br>
		"""
		st.markdown(html_temp, unsafe_allow_html=True)
		st.title("Face Expression WEB Application :")
		st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
		video_file = st.file_uploader("Upload Videos",type=['mp4','mpeg','avi'])
		if video_file is not None:
			tffile = tempfile.NamedTemporaryFile(delete=False)
			tffile.write(video_file.read())


			camera=cv2.VideoCapture(tffile.name) 
			
			while camera.isOpened():
				_, frame= camera.read()
				if not _:
					st.write("done processing....")
					break
				
				frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
				gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
				num_faces=face_cascade.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors=5)
				for x,y,w,h in num_faces:
					cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
					roi_gray_frame = gray_frame[y:y + h, x:x + w]
					cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0) 
					
					emotion_prediction = classifier.predict(cropped_img)
					maxindex = int(np.argmax(emotion_prediction))
					cv2.putText(frame, emotion_labels[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
					stframe.image(frame) 
					

					

					
			
					
					
		
		
		
					




	



		

		
		
		 
	
		 
	 
		
		
		 
		
   
		 





	elif choice == "LiveWebcam":
		html_temp = """
	
			
		<marquee behavior="scroll" direction="left" width="100%;">
		<h2 style= "color: red; font-family: 'Raleway',sans-serif; font-size: 62px; font-weight: 800; line-height: 72px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Face Emotion Recognition System</h2>
		</marquee><br>
		"""
		st.markdown(html_temp, unsafe_allow_html=True)
		
		st.title("Webcam Live Feed")
		st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
		st.subheader(''' 
		Welcome to the other side of the SCREEN!!!
		* Get ready with all the emotions you can express. 
		''')
		st.write("1. Click Start to open your camera and give permission for prediction")
		st.write("2. This will predict your emotion.") 
		st.write("3. When you done, click stop to end.")
		RTC_CONFIGURATION = RTCConfiguration(
				 {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
				) 
		
		webrtc_streamer(key="WYH",mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,media_stream_constraints={"video": True, "audio": False},video_processor_factory=VideoTransformer)  

	elif choice=="About":
		st.title("Facial Emotion Recognition System")
		st.subheader("Emotion Detection Model Using Streamlit & Python")
		st.markdown("This Application Developed by <a href='https://github.com/datamind321'>DataMind Platform 2.0</a>",unsafe_allow_html=True) 
		
		# st.markdown("If You have any queries , Contact Us On : ") 
		st.header("contact us on : bme19rahul.r@invertisuniversity.ac.in")

		st.divider()
		
		st.markdown("[![Linkedin](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/rahul-rathour-402408231/)")
		
		st.markdown("[![Instagram](https://img.icons8.com/color/1x/instagram-new.png)](https://instagram.com/_technical__mind?igshid=YmMyMTA2M2Y=)")



		



	  
				
   

main()	   