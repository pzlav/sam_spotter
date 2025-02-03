import cv2
import time
import os

def record_clip(cap, output_dir, clip_length=3, fps=30):
    """
    Record a short clip (3 seconds by default) from the webcam feed.
    Saves the clip to `output_dir` in .mp4 format.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a unique filename based on current time
    filename = os.path.join(output_dir, f"motion_clip_{int(time.time())}.mp4")
    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
    
    print(f"[INFO] Recording {clip_length}s clip to: {filename}")
    
    # Number of frames to record
    num_frames = int(clip_length * fps)
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    out.release()
    print(f"[INFO] Finished recording: {filename}")

def main():
    # Initialize webcam at index 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open /dev/video0.")
        return
    
    # Optional: set resolution (comment out if you want default resolution)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )
    
    # Directory to save clips
    output_dir = "recorded_clips"
    os.makedirs(output_dir, exist_ok=True)
    
    # Motion detection params
    min_contour_area = 2500  # Adjust as needed
    cooldown = 15  # Wait at least 5 seconds between recordings
    last_record_time = 0
    
    print("[INFO] Starting motion detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame from webcam.")
            break
        
        # Apply background subtraction
        fgmask = fgbg.apply(frame)
        
        # Threshold the mask to remove shadows (which appear gray)
        _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_in_frame = False
        
        # Check if any contour is large enough
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                motion_in_frame = True
                break
        
        # If we detect "significant" motion and not on cooldown, record a clip
        current_time = time.time()
        if motion_in_frame and (current_time - last_record_time > cooldown):
            record_clip(cap, output_dir, clip_length=6, fps=25)
            last_record_time = current_time
        
        # Show frames for debugging (optional)
        cv2.imshow("Webcam", frame)
        cv2.imshow("Motion Mask", thresh)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
