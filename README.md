# Multimodal Kinetic Energy Project

This project analyzes human body movements using **MediaPipe** and computes the **kinetic energy** of different body parts in real time.

---

## Overview
The goal of this project is to explore how computer vision can be used to estimate human motion energy for applications such as **rehabilitation monitoring**, **sports analytics**, or **gesture-based interfaces**.

---

## Technical Details
- Implemented in **Python**
- Used **MediaPipe** for pose detection and keypoint extraction
- Applied basic **physics formulas** to estimate kinetic energy from joint velocity
- Visualized real-time results using **OpenCV**
- Designed for future extension to **multimodal data inputs** (e.g., sensors or video)

---

## Example Output
When the system detects movement through the webcam, it calculates the velocity and kinetic energy for each tracked joint (e.g., wrist, elbow, knee) and overlays them on the video feed in real time.

---

## Future Work
- Integrate accelerometer data for higher accuracy  
- Add personalized energy profiles based on body mass  
- Evaluate real-time performance in different lighting conditions

