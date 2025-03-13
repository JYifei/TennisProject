#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:13:21 2025

@author: komarjo1
"""

from trc import TRCData
import pandas as pd
import numpy as np
import glob



dirDAT = './outputTRC/'
filelist = glob.glob(fr"{dirDAT}/*.trc")

for file in filelist:
        filename = file
        print(filename)

        mocap_data = TRCData()
        mocap_data.load(filename)


        rows=int(mocap_data['NumFrames'])
        num_markers = int(mocap_data['NumMarkers'])
        cols = num_markers*3
        frames = np.zeros((rows, cols+5))


        for i in range(0,int(mocap_data['NumFrames'])):
            #print(i)
            num_frames = mocap_data[i]
            col_vec = np.array(num_frames[1], ndmin=3)
            length=np.shape(col_vec)
            size = length[1]
            x =col_vec[0,range(0,size),0]
            y =col_vec[0,range(0,size),1]
            z =col_vec[0,range(0,size),2]
            numframes = np.array(mocap_data['NumFrames'])
            camerarate = np.array(mocap_data['CameraRate'])
            num_markers = np.array(mocap_data['NumMarkers'])
            time = np.array(num_frames[0])
            frame = np.array(i)+1
            b = np.array((camerarate, num_markers, numframes, time, frame))
            a=np.concatenate((b,x,y,z), axis=0)
            frames[i,range(0,size*3+5)] = a
    
    

        data = pd.DataFrame(frames)
        data.columns = ['CameraRate', 'NumMarkers', 'NumFrames', 'Time', 'Frame',
  'CHip_x','RHip_x','RKnee_x','RAnkle_x','RBigToe_x','RSmallToe_x','RHeel_x','LHip_x','LKnee_x','LAnkle_x','LBigToe_x','LSmallToe_x','LHeel_x','Neck_x','X_Head','Nose_x','RShoulder_x','RElbow_x','RWrist_x','LShoulder_x','LElbow_x','LWrist_x', 
  'CHip_y','YRHip_y','RKnee_y','RAnkle_y','RBigToe_y','RSmallToe_y','RHeel_y','LHip_y','LKnee_y','LAnkle_y','LBigToe_y','LSmallToe_y','LHeel_y','YNeck_y','Y_Head','Nose_y','YRShoulder_y','RElbow_y','RWrist_y','LShoulder_y','LElbow_y','YLWrist_y',
  'CHip_z','RHip_z','RKnee_z','RAnkle_z','RBigToe_z','RSmallToe_z','RHeel_z','LHip_z','LKnee_z','LAnkle_z','LBigToe_z','LSmallToe_z','LHeel_z','Neck_z','Z_Head','Nose_z','RShoulder_z','RElbow_z','RWrist_z','LShoulder_z','ZLElbow_z','LWrist_z']

        data.to_csv((file + '.csv'), mode='w', sep=',', index=False, header=True)  

