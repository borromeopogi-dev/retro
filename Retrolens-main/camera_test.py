import cv2

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if cap.isOpened():
        print(f"Camera found at index {index}")

        ret, frame = cap.read()

        if ret and frame is not None:
            print(f"  Frame received: {frame.shape}")

            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        cap.release()
    else:
        print(f"No camera at index {index}")