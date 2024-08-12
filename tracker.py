import cv2
import numpy as np

def detect_faces(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # detect skin colors
    lower_skin = np.array([0, 20, 70], dtype="uint8")
    upper_skin = np.array([20, 255, 255], dtype="uint8")
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find the contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    faces = []
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            faces.append((x, y, w, h))

    return faces

def main():
    cap = cv2.VideoCapture(0)

    while True:
        # capture fbf (frame by frame)
        ret, frame = cap.read()
        if not ret:
            break

        # faces: i will stalk ur family
        faces = detect_faces(frame)
	
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # draw the scope
            scope_thickness = 2
            scope_size = 40
            cv2.line(frame, (x + w // 2 - scope_size, y + h // 2), (x + w // 2 + scope_size, y + h // 2), (0, 255, 0), scope_thickness)
            cv2.line(frame, (x + w // 2, y + h // 2 - scope_size), (x + w // 2, y + h // 2 + scope_size), (0, 255, 0), scope_thickness)
            cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)

            # display face position
            position_text = f"Position: ({x + w // 2}, {y + h // 2})"
            cv2.putText(frame, position_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('stupid tracker (q to quit)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
