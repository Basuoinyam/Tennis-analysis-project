# Tennis Analysis Project

Analyze tennis matches using artificial intelligence from video, with mini court visualization and detailed player statistics.

---

## Project Idea

This project enables automated analysis of tennis match videos:
- Extract and track the coordinates of players and the ball from the video using computer vision models (YOLO, **ResNet50 Keypoints**).
- Draw a **Mini Court** that shows the positions of players and the ball in real time.
- Calculate statistics for each player (number of shots, speed, distance covered, etc.).

---

## Main File Structure

- **main.py**  
  The main script executes all analysis steps:
  - Read the video.
  - Track players and the ball.
  - Extract court lines (CourtLineDetector) using a trained **ResNet50** model for keypoint prediction.
  - Convert coordinates to the mini court (MiniCourt).
  - Calculate and draw statistics.
  - Output a final video with all analytics.

- **utils/**  
  Helper functions such as reading/writing video, calculating distances and coordinates, drawing statistics on video.

- **mini_court/mini_court.py**  
  Mini court drawing, coordinate conversion for players and the ball, point visualization, and court measurement functions.

- **court_line_detector/court_line_detector.py**  
  Detect court lines and extract keypoints from the first frame of the video using **ResNet50**.

- **trackers/**  
  Track players and the ball using YOLO models, with functions for extracting, filtering, and interpolating detections.

- **training/**  
  Jupyter notebooks for training ball and court keypoint detection models, including training the **ResNet50** model for court keypoints.

- **analysis/ball_analysis.ipynb**  
  Analyze ball data from the video: coordinates, speeds, statistical tables.

---

## How to Run

1. Place your tennis match video in the `input_videos/` folder.
2. Run the main script:
   ```bash
   python main.py
   ```
   The results (the new analytics video) will be saved in the `output_videos/` folder.

---

## Technologies Used

- Python
- Jupyter Notebook
- OpenCV (video processing)
- YOLO (player and ball tracking)
- **ResNet50** (court keypoints training and detection)
- PyTorch (model training)
- Pandas (statistical data analysis)
- Matplotlib or cv2 (visualization of statistics and the mini court)
- Numpay

---

## Example Statistics

- Average shot speed for each player.
- Ball speed between every two shots.
- Player and ball movement visualization on the mini court.
- Statistical tables for every movement or shot.

---

