import cv2
import numpy as np
import mediapipe as mp

# Function to load and preprocess the garment image
def load_garment_image(image_path):
    garment_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if garment_image is None:
        print(f"Error: Failed to load garment image from '{image_path}'.")
        return None, None
    
    if garment_image.shape[2] == 4:
        alpha_channel = garment_image[:, :, 3]
    else:
        alpha_channel = np.ones_like(garment_image[:, :, 0]) * 255  # Fully opaque if no alpha channel
    bgr_channels = garment_image[:, :, :3]
    return bgr_channels, alpha_channel

# Function to perform pose estimation
def perform_pose_estimation(frame):
    with mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is not None:
            return results.pose_landmarks.landmark
        else:
            print("Warning: No pose landmarks detected.")
            return None

# Function to overlay the garment on the frame based on pose landmarks
def try_on_garment(frame, garment_image, alpha_channel, pose_landmarks):
    if pose_landmarks is None:
        print("Error: Cannot overlay garment without pose landmarks.")
        return frame

    # Calculate dimensions for resizing garment image
    height, width = frame.shape[:2]
    garment_height, garment_width = garment_image.shape[:2]

    # Get coordinates for key landmarks
    left_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    # Debug print statements for landmark positions
    print(f"Left Shoulder: ({left_shoulder.x}, {left_shoulder.y})")
    print(f"Right Shoulder: ({right_shoulder.x}, {right_shoulder.y})")
    print(f"Left Hip: ({left_hip.x}, {left_hip.y})")
    print(f"Right Hip: ({right_hip.x}, {right_hip.y})")

    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_distance = np.sqrt((right_shoulder.x - left_shoulder.x) ** 2 + (right_shoulder.y - left_shoulder.y) ** 2)
        torso_height = np.sqrt((left_hip.y - left_shoulder.y) ** 2 + (left_hip.x - left_shoulder.x) ** 2)
        scale_factor = (torso_height * height) / garment_height * 2  # Adjusted scale factor
        print(f"Scale factor: {scale_factor}")
    else:
        scale_factor = 2.0  # Default scale if shoulders or hips not detected

    # Calculate destination size for resized garment
    resized_width = int(garment_width * scale_factor)
    resized_height = int(garment_height * scale_factor)
    print(f"Resized dimensions: {resized_width}x{resized_height}")

    # Ensure resized dimensions are positive and non-zero
    if resized_width <= 0:
        resized_width = 1
    if resized_height <= 0:
        resized_height = 1

    # Resize garment image and alpha channel dynamically
    resized_garment = cv2.resize(garment_image, (resized_width, resized_height))
    resized_alpha_channel = cv2.resize(alpha_channel, (resized_width, resized_height))

    # Coordinates for placing garment image
    top_left_x = int((left_shoulder.x + right_shoulder.x) / 2 * width - resized_garment.shape[1] / 2)
    top_left_y = int(left_shoulder.y * height - resized_garment.shape[0] / 4)  # Adjust the vertical position
    print(f"Top-left corner: ({top_left_x}, {top_left_y})")

    # Ensure garment image stays within frame boundaries
    bottom_right_x = min(top_left_x + resized_garment.shape[1], width)
    bottom_right_y = min(top_left_y + resized_garment.shape[0], height)
    top_left_x = max(top_left_x, 0)
    top_left_y = max(top_left_y, 0)
    print(f"Bounding box: ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})")

    # Overlaying the garment on the frame
    for c in range(frame.shape[2]):
        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x, c] = \
            frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x, c] * \
            (1.0 - resized_alpha_channel[:bottom_right_y - top_left_y, :bottom_right_x - top_left_x] / 255.0) + \
            resized_garment[:bottom_right_y - top_left_y, :bottom_right_x - top_left_x, c] * \
            (resized_alpha_channel[:bottom_right_y - top_left_y, :bottom_right_x - top_left_x] / 255.0)

    return frame

# Main function for real-time try-on process
def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Failed to open webcam.")
        return

    # Load the garment image and preprocess it
    garment_image_path = "image_testing/images__5_-removebg-preview (1).png"
    garment_image, alpha_channel = load_garment_image(garment_image_path)
    if garment_image is None or alpha_channel is None:
        print("Error: Garment image loading failed.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        # Perform pose estimation
        pose_landmarks = perform_pose_estimation(frame)

        if pose_landmarks is not None:
            # Try on the garment
            tryon_image = try_on_garment(frame, garment_image, alpha_channel, pose_landmarks)
        else:
            tryon_image = frame
        
        # Display the result
        cv2.imshow('Virtual Try-On', tryon_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
