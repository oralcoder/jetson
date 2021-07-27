'''
cvlib
Computer Vision library
pip3 install cvlib
pip3 install tensorflow
'''
import numpy as np
import cv2
import cvlib

image = cv2.imread('faceSample.png')
cv2.imshow('image', image)
cv2.waitKey(0)

faces, confidences = cvlib.detect_face(image)

print(faces, confidences)

faceImage = image.copy()

for i in range(len(faces)):
  x1, y1, x2, y2 = faces[i]
  cv2.rectangle(faceImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
  cv2.putText(faceImage, str(confidences[i]), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

cv2.imshow('image', faceImage)
cv2.waitKey(0)

# 성별 인식
genderImage = image.copy()
for i in range(len(faces)):
    # 얼굴 인식 박스 그리기
    x1, y1, x2, y2 = faces[i]
    cv2.rectangle(genderImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(genderImage, str(confidences[i]), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    # ROI 설정 및 성별 인식
    roi = image[y1:y2, x1:x2]
    label, confidence = cvlib.detect_gender(roi)
    print(label, confidence)
    gender = label[np.argmax(confidence)]
    cv2.putText(genderImage, gender, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)

cv2.imshow('image', genderImage)
cv2.waitKey(0)

#모자이크
blurImage = image.copy()
for i in range(len(faces)):
    x1, y1, x2, y2 = faces[i]
    roi = image[y1:y2, x1:x2]
    roiBlurred = cv2.GaussianBlur(roi, ksize=(27,27), sigmaX=0)
    blurImage[y1:y2, x1:x2] = roiBlurred

cv2.imshow('image', blurImage)
cv2.waitKey(0)

cv2.destroyAllWindows()