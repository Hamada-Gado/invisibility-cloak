# importing libraries needed
import time
import logging

import cv2
import numpy as np
from ultralytics import YOLO


def set_logger(name="YOLO"):
    """
    Set the logger for the YOLO model
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)

    return logger


def apply_invisible_cloak(frame, background, result) -> np.ndarray:
    """
    Apply the invisible cloak effect to the frame
    :param frame: The input frame
    :param background: The background frame
    :param result: The result of the YOLO model
    :return: The frame with the invisible cloak effect
    """

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

        # Apply post-processing to the mask
        start = time.perf_counter_ns()
        person_mask = cv2.morphologyEx(
            person_mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8)
        )
        person_mask = cv2.morphologyEx(
            person_mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8)
        )
        person_mask = cv2.medianBlur(person_mask, 5)
        end = time.perf_counter_ns()
        logging.getLogger("YOLO").info(
            f"Mask Post-processing time: {(end - start) * 1e-6:.2f} ms"
        )

        # Invert the mask
        background_mask = cv2.bitwise_not(person_mask)

        # Hide the person
        new_background = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Extract the background
        extracted_background = cv2.bitwise_and(background, background, mask=person_mask)

        # Generating the final output
        frame = cv2.addWeighted(extracted_background, 1, new_background, 1, 0)

    return frame


def draw_fps(frame, start_time, frame_count):
    """
    Draw the FPS on the frame
    :param frame: The input frame
    :param start_time: The start time of the frame
    :param frame_count: The frame count
    :return: The frame with the FPS
    """

    # Calculate the FPS
    fps = frame_count / (time.perf_counter() - start_time)
    fps = f"FPS: {fps:.2f}"

    # Draw the FPS on the frame
    cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 175, 200), 2)

    return frame


def main():
    # setting the logger
    set_logger()

    # capturing the video
    video_capture = cv2.VideoCapture(1)
    video_output = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
    )

    # making an object of YOLO
    model = YOLO("yolov8s-seg.pt")

    # capturing the background
    _, background = video_capture.read()
    for _ in range(30):
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
        start_time = time.perf_counter()
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
        elif key == ord("q"):
            break
        elif key == ord("r"):
            for _ in range(5):
                _, background = video_capture.read()
            background = cv2.flip(background, 1)

        # Apply the invisible cloak
        if inv_cloak:
            result = result or model.track(frame, persist=True, half=True)[0]
            frame = apply_invisible_cloak(frame, background, result)

            #! still experimenting with this, DON'T UNCOMMENT
            # It works better with bigger models, but still may need some post-processing
            # background = frame.copy()

        # Show the segmented output
        if show_segmented:
            result = result or model.track(frame, persist=True, half=True)[0]
            frame = result.plot(img=frame)

        # Draw the FPS
        if show_fps:
            frame = draw_fps(frame, start_time, 1)

        cv2.imshow("Dumbledore's Army", frame)
        video_output.write(frame)

    video_capture.release()
    cv2.destroyAllWindows()
    video_output.release()


if __name__ == "__main__":
    main()
