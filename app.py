import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase,WebRtcMode, RTCConfiguration
import av
from pose_utils import tracker
import os
from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
# account_sid = 'ACa8c8a3b595528eaaf5aebc2547600f81'
# auth_token = 'a849ad3fa7bae833be94c77c2b91d94a'
# ICE_SERVERS = Client(account_sid, auth_token).tokens.create().ice_servers

exercises = ["bicep","dumbbell_rows"]
selected_exercise = st.selectbox("Choose an exercise:", exercises)

class PoseEstimationProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="rgb24")
        image = cv2.flip(image, 1)
        try:
            image = tracker(image,selected_exercise)
        except Exception as e:
            print(e)
            pass
        return av.VideoFrame.from_ndarray(image, format="rgb24")

st.title("Pose Estimation App")
#RTC_CONFIGURATION = RTCConfiguration({"iceServers": ICE_SERVERS})
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
webrtc_streamer(
    key="pose-estimation",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=PoseEstimationProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
