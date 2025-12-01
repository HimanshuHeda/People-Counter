# People Counter Project

## Overview
This project uses computer vision and deep learning to count people moving through a video. It leverages the YOLOv8 object detection model and the SORT tracking algorithm to detect, track, and count people crossing predefined lines in a video frame.

## Features
- Detects people in video frames using YOLOv8
- Tracks individuals across frames with SORT
- Counts people moving up and down across specified lines
- Visualizes bounding boxes, IDs, and counts on the video

## Requirements
- Python 3.8+
- OpenCV
- cvzone
- ultralytics (YOLO)
- numpy
- filterpy
- scikit-image
- matplotlib
- scipy

## Installation
1. Clone this repository or copy the project files.
2. Install dependencies:
   ```bash
   pip install opencv-python cvzone ultralytics numpy filterpy scikit-image matplotlib scipy
   ```
3. Place your video file in the `Videos` folder (e.g., `people.mp4`).
4. Ensure you have `mask.png` and `graphics.png` in the project directory.

## Usage
Run the project using the correct Python executable:
```bash
C:/Users/himan/AppData/Local/Programs/Python/Python313/python.exe People_Counter.py
```

## How It Works
- The script loads a video and processes each frame.
- A mask is applied to focus detection on a region of interest.
- YOLOv8 detects people in the frame.
- SORT tracks detected people and assigns unique IDs.
- When a person crosses the up or down line, their ID is counted.
- The results are displayed in real-time with bounding boxes and counts.

## Model Details
- **YOLOv8**: Used for fast, accurate object detection. The model file `yolov8l.pt` is loaded from the `Yolo-Weights` directory.
- **SORT**: A simple, online, and real-time tracking algorithm implemented in `sort.py`.

## Customization
- To speed up detection, use a smaller YOLO model (e.g., `yolov8n.pt`).
- Adjust the line coordinates in `limitsUp` and `limitsDown` for your scenario.

## Troubleshooting
- If the video window closes immediately, check your video path and ensure all required files exist.
- Always run the script with the correct Python environment where dependencies are installed.

## Credits
- YOLOv8 by Ultralytics
- SORT by Alex Bewley
- OpenCV and cvzone for visualization

## License
This project is for educational purposes. See individual libraries for their licenses.
