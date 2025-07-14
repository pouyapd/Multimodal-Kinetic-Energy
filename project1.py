import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to compute kinetic energy for body movements


def compute_kinetic_energy(landmarks, velocity_threshold=5, energy_threshold=0.1, movement_threshold=1e-3):
    if len(landmarks) < 2:
        return 0
    energies = []
    for i in range(len(landmarks) - 1):
        dx = landmarks[i+1][0] - landmarks[i][0]
        dy = landmarks[i+1][1] - landmarks[i][1]
        dz = landmarks[i+1][2] - landmarks[i][2]
        # Filter out very small movements
        if abs(dx) < movement_threshold and abs(dy) < movement_threshold and abs(dz) < movement_threshold:
            continue  # Skip small movements
        velocity = np.sqrt(dx**2 + dy**2 + dz**2)
        if velocity > velocity_threshold:
            energy = 0.5 * velocity**2
            if energy > energy_threshold:
                energies.append(energy)
    return np.mean(energies) if energies else 0

# Function to apply moving average filter


def apply_moving_average(values, window_size=5):
    if len(values) < window_size:
        return np.mean(values)
    return np.mean(values[-window_size:])


# Use queues to store recent landmark data for each body part
landmark_queues = {
    'Whole Body': deque(maxlen=10),
    'Upper Body': deque(maxlen=10),
    'Lower Body': deque(maxlen=10),
    'Right Arm': deque(maxlen=10),
    'Left Arm': deque(maxlen=10),
    'Right Leg': deque(maxlen=10),
    'Left Leg': deque(maxlen=10)
}

# Capture video from webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose", 1280, 720)

plt.ion()  # Enable interactive mode for live plotting

# Variables to store the report data
energy_report = {part: [] for part in landmark_queues.keys()}
x_data = []  # For the x-axis (time or frame index)
# For the y-axis (energy values)
y_data = {part: [] for part in landmark_queues.keys()}

try:
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape
            for i, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                landmark_queues['Whole Body'].append((cx, cy, cz))
                # Assign landmarks to subsets based on their indices
                if i in range(11, 23):  # Upper body
                    landmark_queues['Upper Body'].append((cx, cy, cz))
                if i in range(23, 25):  # Lower body
                    landmark_queues['Lower Body'].append((cx, cy, cz))
                if i in range(11, 17):  # Right arm
                    landmark_queues['Right Arm'].append((cx, cy, cz))
                if i in range(17, 23):  # Left arm
                    landmark_queues['Left Arm'].append((cx, cy, cz))
                if i in range(25, 29):  # Right leg
                    landmark_queues['Right Leg'].append((cx, cy, cz))
                if i in range(29, 33):  # Left leg
                    landmark_queues['Left Leg'].append((cx, cy, cz))

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Compute kinetic energy for each subset
            energies = {
                part: compute_kinetic_energy(landmark_queues[part])
                for part in landmark_queues.keys()
            }

            # Apply moving average to smooth out energy values
            smoothed_energies = {
                part: apply_moving_average(
                    energy_report[part] + [energies[part]])
                for part in energies.keys()
            }

            # Save smoothed energies to the report
            for part, energy in smoothed_energies.items():
                energy_report[part].append(energy)

            # Add the current frame count to x_data (time)
            x_data.append(frame_count)
            for part, energy in smoothed_energies.items():
                y_data[part].append(energy)

            # Plot the kinetic energy distribution in real-time (Line plot)
            plt.clf()  # Clear the previous plot
            for part in y_data:
                plt.plot(x_data, y_data[part], label=part)
            plt.xlabel('Time (Frames)')
            plt.ylabel('Kinetic Energy')
            plt.title('Kinetic Energy Distribution Over Time')
            plt.legend()
            plt.pause(0.001)  # Pause to update the plot

        # Display the frame with landmarks
        cv2.imshow("Pose", frame)

        # Check for exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.close()

    # Generate final energy report
    print("\nKinetic Energy Report (End of Session):")
    with open("energy_report.csv", "w") as file:
        file.write("Body Part, Average Energy\n")
        for part, energies in energy_report.items():
            avg_energy = np.mean(energies)
            print(f"{part}: {avg_energy:.4f}")
            file.write(f"{part}, {avg_energy:.4f}\n")
