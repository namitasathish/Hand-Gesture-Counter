import cv2
import os
import time
import HandTrackingModule as htm

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = os.path.join(os.path.dirname(__file__), 'images')
myList = os.listdir(folderPath)
print(myList)

overlayList = []

for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    image = cv2.resize(image, (300, 300))  
    overlayList.append(image)

print(f'Loaded {len(overlayList)} images.')

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds=[4,8,12,16,20]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from webcam.")
        break

    img = detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    #print(lmList)

    if len(lmList)!=0:
        fingers=[]
        

        #for right thumb finger
        if lmList[tipIds[0]][1]>lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            

           

        #for other fingers
        for id in range(1,5):
          if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
            fingers.append(1)
          else:
              fingers.append(0)


        #print(fingers)
        totalFingers=fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[0].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]
        #cv2.rectangle(img, (80, 325), (230, 525), (0, 0, 0), cv2.FILLED)


        cv2.putText(img, str(totalFingers), (110, 475), cv2.FONT_HERSHEY_PLAIN,
                                                            10, (255,0,255),15)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (1000, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
