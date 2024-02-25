# importing libraries needed
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


def apply_invisible_cloak(frame, background, result) -> np.ndarray:
    ###? hmmmm returning the frame instead of the background doesn't work

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
        back = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Extract the background
        fore = cv2.bitwise_and(background, background, mask=person_mask)

        # Generating the final output
        frame = cv2.addWeighted(back, 1, fore, 1, 0)

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

    # The Main loop
    inv_cloak = False
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Getting the output from YOLO
        result = model.predict(frame, half=True)[0]

        # Show the segmented output
        key = cv2.waitKey(25)
        if key == ord("s"):
            frame = result.plot()

        # Apply the invisible cloak
        if key == ord("c"):
            inv_cloak = not inv_cloak

        if inv_cloak:
            frame = apply_invisible_cloak(frame, background, result)

        cv2.imshow("Thankyou_Dumbledore", frame)
        if key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
