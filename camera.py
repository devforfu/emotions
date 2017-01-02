import cv2


window_id = "Camera"
capture = cv2.VideoCapture(0)
cv2.namedWindow(window_id, cv2.WINDOW_AUTOSIZE)

while True:
    grabbed, frame = capture.read()
    if not grabbed:
        break

    frame = cv2.pyrDown(frame)
    frame = cv2.pyrDown(frame)
    cv2.imshow(window_id, frame)
    key = cv2.waitKey(33)

    if key == 27:
        break

cv2.destroyWindow(window_id)


