import cv2
import numpy as np

def real_time_face_detection():
    cap = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('facial-detection/opencvXML/haarcascade_frontalface_default.xml')

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype="uint8")
        faces = haar.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Live Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame, faces

# Main execution flow
def main():
    final_frame, detected_faces = real_time_face_detection()
    print(f"Detected {len(detected_faces)} faces in the final frame")
    return detected_faces


# Run the main function
if __name__ == "__main__":
    main()