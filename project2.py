import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import dequeq

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
            continue
        velocity = np.sqrt(dx**2 + dy**2 + dz**2)
        if velocity > velocity_threshold:
            energy = 0.5 * velocity**2
            if energy > energy_threshold:
                energies.append(energy)
    return np.mean(energies) if energies else 0


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

# Initialize interactive plot
plt.ion()
fig, ax = plt.subplots()
ax.set_ylabel('Kinetic Energy')
ax.set_title('Real-time Kinetic Energy Distribution')
labels = list(landmark_queues.keys())
bars = ax.bar(labels, [0] * len(labels))

# Variables to store the report data
energy_report = {part: [] for part in landmark_queues.keys()}
frame_counter = 0  # Counter to control frequency of data update

try:
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

            # Compute kinetic energy for each subset only after a certain number of frames
            # Process data every 5 frames (adjust as needed)
            if frame_counter % 5 == 0:
                energies = {
                    part: compute_kinetic_energy(landmark_queues[part])
                    for part in landmark_queues.keys()
                }

                # Save energies to the report
                for part, energy in energies.items():
                    energy_report[part].append(energy)

                # Update real-time plot
                for bar, value in zip(bars, energies.values()):
                    bar.set_height(value)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

                # Analyze dominant and least active movements
                total_energy = sum(energies.values())
                if total_energy > 0:
                    max_part = max(energies, key=energies.get)
                    min_part = min(energies, key=energies.get)
                    print(f"Most active: {
                          max_part} | Least active: {min_part}")

            frame_counter += 1  # Increment frame counter

        # Display the frame with landmarks
        cv2.imshow("Pose", frame)

        # Check for exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Generate final energy report
    print("\nFinal Kinetic Energy Report:")
    with open("energy_report.txt", "w") as file:
        final_energies = {part: np.mean(values)
                          for part, values in energy_report.items()}
        for part, avg_energy in final_energies.items():
            print(f"{part}: {avg_energy:.4f}")
            file.write(f"{part}: {avg_energy:.4f}\n")

    # Display final energy distribution chart
    plt.ioff()
    plt.clf()
    labels = list(final_energies.keys())
    values = list(final_energies.values())

    plt.bar(labels, values, color='skyblue')
    plt.ylabel('Kinetic Energy')
    plt.title('Final Kinetic Energy Distribution')
    plt.show()
