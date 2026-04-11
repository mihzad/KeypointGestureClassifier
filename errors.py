import cv2
import sys

print("--- Diagnostic Start ---")

# 1. Try to open the camera
camera_index = 0 # Try 0, 1, or -1 if 0 doesn't work
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"ERROR: Could not open camera at index {camera_index}.")
    sys.exit()
else:
    print(f"SUCCESS: Camera {camera_index} found and opened.")

# 2. Try to read a single frame
success, frame = cap.read()
if not success:
    print("ERROR: Camera opened, but could not read a frame (is another app using it?)")
    cap.release()
    sys.exit()
else:
    print("SUCCESS: Frame captured successfully.")

# 3. Try to open a window
print("Attempting to open window... Press 'q' to close.")
while True:
    success, frame = cap.read()
    if not success:
        break
    
    cv2.imshow('Diagnostic Window', frame)
    
    # Check for 'q' key or window close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("--- Diagnostic End ---")