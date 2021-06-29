import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam= 648, 488
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "fingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIDs = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if len(lmList) !=0:
        fingers = []

        #thumb
        if lmList[tipIDs[0]][1] > lmList[tipIDs[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #fourfingers
        for id in range(1,5):
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print fingers
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w, 0:c] = overlayList[totalFingers]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}]',(450, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
