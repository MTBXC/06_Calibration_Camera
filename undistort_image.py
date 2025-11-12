import numpy as np
import cv2
import os

# --- CONFIGURATION ---
PARAMS_FILE = 'camera_params.npz'
# Choose Random image to test it out.
TEST_IMAGE_PATH = os.path.join('calib_images', 'calib_image_10.png')
OUTPUT_IMAGE_PATH = 'corrected_image.png'

# --- Main function ---
def undistort_and_display(image_path, params_file):
    """Loads calibration parameters, corrects the distortion of an image, and displays it."""
    
    # 1. Load calibration parameters
    try:
        with np.load(params_file) as X:
            mtx, dist = [X[i] for i in ('mtx', 'dist')]
    except FileNotFoundError:
        print(f"Error: Calibration file '{params_file}' not found.")
        return
        
    # 2. Load the image to correct
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Test image '{image_path}' not found.")
        return

    h, w = img.shape[:2]

    # 3. Calculate optimized camera matrix
    # getOptimalNewCameraMatrix adjusts the camera matrix to minimize black border after correction
    # alpha=1.0 retains all pixels (some black borders), alpha=0.0 crops the image to no black borders
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 4. Correct the image (Undistortion)
    # Metoda remap jest lepsza jakościowo
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # 5. Crop the image based on ROI (Region of Interest)
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w] # Opcjonalne przycięcie, jeśli alpha=0.0

    # 6. Save and display the result
    cv2.imwrite(OUTPUT_IMAGE_PATH, dst)
    print(f"\nCorrected image saved to: {OUTPUT_IMAGE_PATH}")
    
    # Displaying the results (may fail due to GTK error, use NoMachine to view saved file if necessary)
    try:
        # Resize images to fit on screen if they are too large (1920x1080)
        resized_original = cv2.resize(img, (w // 2, h // 2))
        resized_corrected = cv2.resize(dst, (w // 2, h // 2))

        cv2.imshow('01 - Original Image', resized_original)
        cv2.imshow('02 - Corrected Image', resized_corrected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Warning: Could not display images (GTK error). Please check the 'corrected_image.png' file.")


if __name__ == "__main__":
    undistort_and_display(TEST_IMAGE_PATH, PARAMS_FILE)