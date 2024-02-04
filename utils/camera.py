import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os 

class Camera():
    def __init__(self):
        
        #configure color 
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        #Get device product line for setting supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with color sensor support")
            exit(0)
            
        self.config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
        
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color,960,540,rs.format.bgr8,30)
        else:
            self.config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
            
        #start streaming
        self.pipeline.start(self.config)
        
    def get_rgb(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        color_image = np.asanyarray(color_frame.get_data())
        return color_image[50:250,250:450]