
import cv2 
import numpy as np 
import torch 
from models import NaimishNet, ResNet18
import argparse
import os 

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="Path to weights file")
ap.add_argument("-m", "--model", choices=['naimish', 'resnet'], required=True, help="Select Model")
ap.add_argument("-c", "--cam", required=True, help="Source for camera feed")

args = ap.parse_args()

model_weights = args.weights
model_name = args.model
cam_src = args.cam

if args.model_name == "naimish":
    net = NaimishNet()
elif args.model_name == "resnet":
    net = ResNet18()
else:
    print("Invalid model")

img_size = 224
net.load_state_dict(torch.load(model_weights))

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

def detect_keypoints(self, img):
    #Detecting faces in an image
    faces = face_cascade.detectMultiScale(img, 1.2, 2)

    if len(faces) < 0: return img

    for (x,y,w,h) in faces:
        # don't scale ouside of the frame!
        if (y-scale) < 0 and (x-scale) < 0:
            if (y-scale) < (x-scale):
                scale += (y-scale)
            else:
                scale += (x-scale)
        elif (y-scale) < 0 :
            scale += (y-scale)
        elif (x-scale) < 0:
            scale += (x-scale)

        face_roi = img[y-scale:(y+h-scale), x-scale:(x+w-scale)]

        face_roi_bgr = face_roi.copy()

        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        face_roi_gray/=255.0

        h, w = face_roi_gray.shape[:2]
        face_roi_gray = cv2.resize(face_roi_gray, (img_size, img_size))
        dh, dw = face_roi_gray.shape[:2]

        scalling_factor_x = h/dh
        scalling_factor_y = w/dw

        # if image has no grayscale color channel, add one
        if(len(face_roi_gray.shape) == 2):
            # add that third color dim 
            face_roi_gray = face_roi_gray[..., np.newaxis]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        face_roi_gray = face_roi_gray.transpose((2, 0, 1))
        face_roi_gray = torch.from_numpy(face_roi_gray)

        face_roi_gray.unsqueeze_(0) #Add 1 dim for one batch  B X C X H X W 

        keypoints = net(face_roi_gray)

        keypoints = keypoints.view(68, -1) # (68, 2)
        keypoints = keypoints.detach().numpy()

        # undo normalization of keypoints
        keypoints = keypoints*(face_roi_gray.shape[0]/4)+face_roi_gray.shape[0]/2
        for pts in keypoints:
          pts[0] = x-scale +pts[0] * scalling_factor_x
          pts[1] = y-scale +pts[1] * scalling_factor_y

        color = (0, 255, 0)
        

        for i in range(len(keypoints)):
            if (i != 16 and i != 21 and i != 26 and i != 30 and i != 35 and i < 68) :
                pt1 = (keypoints[i][0], keypoints[i][1])
                
                #Coordinates can be refered to data visualization on notebook (1.)
                if i == 17:
                    # left eyebrow
                    color = (0,100,0)
                elif i == 22:
                    # right eyebrow
                    color = (0,100,0)
                elif i == 27:
                    # nose stem
                    color = (255,255,0)
                elif i == 31:
                    # nose tip
                    color = (255,255,0)
                elif i == 36:
                    # left eye
                    color = (0,250,154)
                elif i == 42:
                    # right eye
                    color = (0,250,154)
                elif i == 48:
                    # lips
                    color = (255,20,147)
                     
                if i == 41:
                    pt2 = (keypoints[36][0], keypoints[36][1]) 
                elif i == 47:
                    pt2 = (keypoints[42][0], keypoints[42][1])
                elif i == 67:
                    pt2 = (keypoints[60][0], keypoints[60][1])
                else:
                    pt2 = (keypoints[i+1][0], keypoints[i+1][1])
                    
                cv2.line(face_roi_bgr, pt1, pt2, color, thickness=5, lineType=8, shift=0) 
                
        return face_roi_bgr


if name == '__main__':
    cap = cv2.videoCapture(cam_src)

    while cap.isOpened():
      ret, frame = cap.read()
      
      if ret: 
        detected_keypts = detect_keypoints(frame) 
        detected_keypts = cv2.cvtColor(detected_keypts, cv2.COLOR_RGB2BGR)
        cv2.imshow("feed", detected_keypts)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
      else:
        break

    print("[INFO] Ending stream..")
    cv2.destroyAllWindows()
    cap.release()
