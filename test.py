import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *


model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('https://cctv1.dreamnet.su:8090/hls/275779/8c728f28f72aea02c41d/playlist.m3u8')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count=0

area=[(467,265),(411,272),(718,348),(732,319)]
tracker=Tracker()
area_c=set()
while True:
    
    ret,frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))
    if ret is None:
        break
    count += 1
    if count % 3 != 0:
        continue

    results=model.predict(frame)


    a=results[0].boxes.boxes

    px=pd.DataFrame(a).astype("float")
    #print(px)
    list=[]
    for index,row in px.iterrows():
  #      print(row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        results=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if results >=0:
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
            cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(225,0,0),2)
            area_c.add(id)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,0),3)
    count=len(area_c)
    cv2.putText(frame, str(count), (61, 146), cv2.FONT_HERSHEY_PLAIN,5, (225, 255, 255), 2)
    with open("count.txt", "a") as file:
        file.write(str(count))
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
