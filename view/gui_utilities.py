import cv2


def capture_image():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
    
        # Display the frame
        cv2.imshow('Webcam', frame)

        # Check if the 's' key is pressed to save the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('captured_image.jpg', frame)
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

