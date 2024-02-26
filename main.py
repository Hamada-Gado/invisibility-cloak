# importing libraries needed
import time

import cv2
import numpy as np
from ultralytics import YOLO


def apply_invisible_cloak(frame, background, result) -> np.ndarray:
    # If there is no detection, return the original frame
    if result.masks is None:
        return frame

    # Extracting the boxes and masks
    data = [
        (box, mask)
        for box, mask in zip(result.boxes, result.masks.data)
        if box.cls == 0
    ]

    # If there are no people, return the original frame
    if len(data) == 0:
        return frame

    for box, mask in data:
        # If the object is not a person, skip it
        if box.cls != 0:
            continue

        # Convert the mask to a format that OpenCV can use
        person_mask = mask.cpu().numpy().astype(np.uint8) * 255
        person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]))

        # Invert the mask
        background_mask = cv2.bitwise_not(person_mask)

        # Hide the person
        new_background = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Extract the background
        extracted_background = cv2.bitwise_and(background, background, mask=person_mask)

        # Generating the final output
        frame = cv2.add(extracted_background, new_background)

    return frame


def draw_fps(frame, start_time, frame_count):
    # Calculate the FPS
    fps = frame_count / (time.time() - start_time)
    fps = f"FPS: {fps:.2f}"

    # Draw the FPS on the frame
    cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def main():
    # capturing the video
    video_capture = cv2.VideoCapture(1)

    # making an object of YOLO
    model = YOLO("yolov8n-seg.pt")

    # let the camera warm up
    time.sleep(3)

    # capturing the background
    _, background = video_capture.read()
    for _ in range(45):
        _, background = video_capture.read()
    background = cv2.flip(background, 1)

    # The Main loop
    inv_cloak = False
    show_fps = False
    show_segmented = False
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        result = None
        start_time = time.time()
        frame = cv2.flip(frame, 1)

        # Getting the key pressed
        key = cv2.waitKey(25)

        # Handling the key presses
        if key == ord("c"):
            inv_cloak = not inv_cloak
        elif key == ord("f"):
            show_fps = not show_fps
        elif key == ord("s"):
            show_segmented = not show_segmented

        # Apply the invisible cloak
        if inv_cloak:
            result = result or model.predict(frame, half=True)[0]
            frame = apply_invisible_cloak(frame, background, result)
            # background = frame.copy() #! still experimenting with this

        # Show the segmented output
        if show_segmented:
            result = result or model.predict(frame, half=True)[0]
            frame = result.plot(img=frame)

        # Draw the FPS
        if show_fps:
            frame = draw_fps(frame, start_time, 1)

        cv2.imshow("Dumbledore's Army", frame)
        if key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
