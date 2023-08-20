import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase,WebRtcMode, RTCConfiguration
import av
from pose_utils import tracker
class PoseEstimationProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        try:
            image = tracker(image)
        except Exception as e:
            print(e)
            pass
        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Pose Estimation App")
RTC_CONFIGURATION = RTCConfiguration(
{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
webrtc_streamer(
    key="pose-estimation",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=PoseEstimationProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)