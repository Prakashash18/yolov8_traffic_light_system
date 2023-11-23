import cv2

cap = cv2.VideoCapture('rtsp://192.168.0.151/rtsp/streaming?channel=1&subtype=0')

#rtsp://admin:admin1234@192.168.0.121:554/live2

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# curl -v -X DESCRIBE "rtsp://admin:admin1234@192.168.1.11:554/Streaming/Channels/101" RTSP/1.0
