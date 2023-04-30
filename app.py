import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
from keras.models import load_model,model_from_json
from model.model import FacialExpressionModel
import time
from bokeh.models.widgets import Div
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode




#importing the cnn model+using the CascadeClassifier to use features at once to check if a window is not a face region
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier("facerecog/haarcascade_frontal_face_default.xml")
model = FacialExpressionModel("model/model_cnn_emotion.json", "model/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX 

#face exp detecting function
def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:

			fc = gray[y:y+h, x:x+w]
			roi = cv2.resize(fc, (48, 48))
			pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
			cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
			return img,faces,pred 
#the main function	 




	
    











# EMOTIONS_LIST = ["Angry", "Disgust",
#                      "Fear", "Happy",
#                      "Neutral", "Sad",
#                      "Surprise"]

# emotion_dict = {0:'Angry', 1 :'Disgust', 2: 'Fear', 3:'Happy', 4: 'Neutral',5:'Sad',6:'Surprise'}
# emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise'] 


classifier =load_model('model/model_78.h5')

# load weights into new model
classifier.load_weights("model/model_weights_78.h5")

# json_file = open('model/emotion_mode.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# classifier = model_from_json(loaded_model_json)

# load weights into new model
# classifier.load_weights("model/emotion_model1.h5")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        RTC_CONFIGURATION = RTCConfiguration(
                 {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
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
	
	

	activities = ["Home","Images" ,"LiveWebcam","About"]
	choice = st.sidebar.selectbox("Select Activity",activities)

	if choice == 'Home': 
		html_temp = """
			
		<marquee behavior="scroll" direction="left" width="100%;">
		<h2 style= "color: white; font-family: 'Raleway',sans-serif; font-size: 62px; font-weight: 800; line-height: 72px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Try your own test! </h2>
		</marquee><br>
		"""
		st.markdown(html_temp, unsafe_allow_html=True)
		
		st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
		
		
	
	
	if choice == 'Images':
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
				if st.image(result_img) :
					st.success("Found {} faces".format(len(result_faces)))

					if prediction == 'Happy':
						st.subheader("YeeY!  You are Happy :smile: today , Always Be ! ")
						
						
					elif prediction == 'Angry':
						st.subheader("You seem to be angry :rage: today ,Take it easy! ")
						
						
					elif prediction == 'Disgust':
						st.subheader("You seem to be Disgust :rage: today! ")
						
						
					elif prediction == 'Fear':
						st.subheader("You seem to be Fearful :fearful: today ,Be couragous! ")
						
						
					elif prediction == 'Neutral':
						st.subheader("You seem to be Neutral today ,Happy day! ")
						
						
						
					elif prediction == 'Sad':
						st.subheader("You seem to be Sad :sad: today ,Smile and be happy! ")
						
						
					elif prediction == 'Surprise':
						st.subheader("You seem to be surprised today ! ")
					
						
					else :
						st.error("Your image does not match the training dataset's images! Try an other image!")
						
    



		

		
		
         
	
		 
	 
		
		
		 
		
   
         





	elif choice == "LiveWebcam":
		st.header("Webcam Live Feed")
		st.subheader('''
        Welcome to the other side of the SCREEN!!!
        * Get ready with all the emotions you can express. 
        ''')
		st.write("1. Click Start to open your camera and give permission for prediction")
		st.write("2. This will predict your emotion.") 
		st.write("3. When you done, click stop to end.")
		
		webrtc_streamer(key="example",mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,media_stream_constraints={"video": True, "audio": False},video_processor_factory=VideoTransformer)  

	elif choice=="About":
		st.write("This Application Developed by DataMind Platform 2.0") 
		st.subheader("Emotion Detection Model Using Streamlit & Python")
		# st.markdown("If You have any queries , Contact Us On : ") 
		st.header("contact us on Email Id :  rahul.r12datascientist@gmail.com")
		
		st.markdown("[![Linkedin](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/rahul-rathour-402408231/)")
		st.markdown("[![Github](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/datamind321)")
		st.markdown("[![Instagram](https://img.icons8.com/color/1x/instagram-new.png)](https://instagram.com/_technical__mind?igshid=YmMyMTA2M2Y=))")



	    



      
				
   

main()	   
