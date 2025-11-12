import cv2
import time
import os

CAMERA_INDEX = 0

#Use Camera from CSI source by GSSTREAMER
GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id=%d ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
) % CAMERA_INDEX
CAPTURE_SOURCE = GSTREAMER_PIPELINE

OUTPUT_FOLDER = "calib_images"

OUTPUT_FOLDER = "calib_images"

# --- Main function ---
def capture_calibration_images(source=CAPTURE_SOURCE):
    """Captures images from the camera and saves them upon pressing 's'."""
    
    # 1. Check/Create the output directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created directory: {OUTPUT_FOLDER}")

    # 2. Camera initialization
    if isinstance(source, str):
        # Use GStreamer string
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open camera. Check the index (0, 1, ...) or GStreamer string.")
        return

    print("\n--- Starting Capture ---")
    print(f"Save location: {OUTPUT_FOLDER}/")
    print("-> Press 's' to SAVE the image.")
    print("-> Press 'q' to QUIT.")

    img_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            time.sleep(1)
            continue
        
        # Optional: display the count of currently saved images
        text = f"Saved: {img_count} images"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera - Calibration', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            filename = os.path.join(OUTPUT_FOLDER, f'calib_image_{img_count:02d}.png')
            cv2.imwrite(filename, frame)
            print(f"✅ Saved: {filename}")
            img_count += 1
            time.sleep(0.5) # Short pause to prevent accidental double saves
        
        elif key == ord('q'):
            break

    # 3. Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Capture Finished ---")


if __name__ == "__main__":
    capture_calibration_images()