from utils import (read_video, save_video,measure_distance,convert_pixel_distance_to_meters,draw_player_stats)
from trackers import PlayerTracker , BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import constants
import cv2
import pandas as pd
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

def main():
    #read video
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)

    # detect players and Ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolo5_best.pt")

    player_detections = player_tracker.detect_frames(video_frames , read_from_stub=True ,
                                                     stub_path="tracker_stubs/player_detections.pkl")

    ball_detections = ball_tracker.detect_frames(video_frames , read_from_stub=True ,
                                                     stub_path="tracker_stubs/ball_detections.pkl")

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

     # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose player
    player_detections = player_tracker.choose_and_filter_players(court_keypoints ,player_detections)

    # mini court
    mini_court = MiniCourt(video_frames[0])
    #detect ball shots
    ball_shots_frames=ball_tracker.get_ball_shot_frames(ball_detections)
    #convert position to mini court position
    ball_mini_court_position,player_mini_court_position=mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections,
        ball_detections,
        court_keypoints,
    )
    #calc speed of ball
    player_stats_data=[{
        "frame_num":0,
        "player_1_number_of_shoots":0,
        "player_1_total_shots_speed": 0,
        "player_1_last_shot_speed": 0,
        "player_1_total_player_speed": 0,
        "player_1_last_player_speed": 0,
        "player_2_number_of_shoots": 0,
        "player_2_total_shots_speed": 0,
        "player_2_last_shot_speed": 0,
        "player_2_total_player_speed": 0,
        "player_2_last_player_speed": 0,
    }]
    for ball_shot_index in range(len(ball_shots_frames) - 1):
        start_frame = ball_shots_frames[ball_shot_index]
        end_frame = ball_shots_frames[ball_shot_index + 1]
        ball_shot_time_in_seconds = (start_frame - end_frame) / 24
        distance_covered_by_ball_in_pixels = measure_distance(ball_mini_court_position[start_frame][1],
                                                              ball_mini_court_position[end_frame][1])
        distance_covered_by_ball_in_meters=convert_pixel_distance_to_meters(
            distance_covered_by_ball_in_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_of_ball_shots=distance_covered_by_ball_in_meters/ball_shot_time_in_seconds * 3.6
    #calc speed of player
        player_position=player_mini_court_position[start_frame]
        player_shot_ball=min(player_position.keys(), key= lambda k: measure_distance(
            player_position[k],
            ball_mini_court_position[start_frame][1],
        ))
        opponent_player_id=1 if player_shot_ball==2 else 2
        distance_covered_by_player_in_pixels = measure_distance(
        player_mini_court_position[start_frame][opponent_player_id],
        ball_mini_court_position[end_frame][1])
        distance_covered_by_player_in_meters=convert_pixel_distance_to_meters(distance_covered_by_player_in_pixels,
                                                                          constants.DOUBLE_LINE_WIDTH,
                                                                          mini_court.get_width_of_mini_court())
        speed_of_opponent=distance_covered_by_player_in_meters/ball_shot_time_in_seconds * 3.6
        current_player_stats=deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"]=start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shoots"]+=1
        current_player_stats[f"player_{player_shot_ball}_total_shots_speed"]+=speed_of_ball_shots
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"]=speed_of_ball_shots
        current_player_stats[f"player_{opponent_player_id}_total_player_speed"]+=speed_of_opponent
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"]=speed_of_opponent
        player_stats_data.append(current_player_stats)
    player_stats_data_df=pd.DataFrame(player_stats_data)
    frames_df=pd.DataFrame({"frame_num":list(range(len(video_frames)))})
    player_stats_data_df=pd.merge(frames_df,player_stats_data_df,on="frame_num",how="left")
    player_stats_data_df=player_stats_data_df.ffill()
    player_stats_data_df["player_1_average_shot_speed"]=player_stats_data_df["player_1_total_shots_speed"]/player_stats_data_df["player_1_number_of_shoots"]
    player_stats_data_df["player_2_average_shot_speed"] = player_stats_data_df["player_2_total_shots_speed"] / \
                                                          player_stats_data_df["player_2_number_of_shoots"]
    player_stats_data_df["player_1_average_player_speed"]=player_stats_data_df["player_1_total_player_speed"] / player_stats_data_df["player_1_number_of_shoots"]
    player_stats_data_df["player_2_average_player_speed"] = player_stats_data_df["player_2_total_player_speed"] / \
                                                            player_stats_data_df["player_2_number_of_shoots"]
    # draw bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames,ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    #Drow mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames=mini_court.drow_point_in_mini_court(output_video_frames,ball_mini_court_position)
    output_video_frames=mini_court.drow_point_in_mini_court(output_video_frames,player_mini_court_position,color=(255,0,0))
    #Drow player stats
    output_video_frames=draw_player_stats(output_video_frames,player_stats_data_df)

    #Draw frame number on top left
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    save_video(output_video_frames,f"output_videos/output_video4.avi")


if __name__ == "__main__":
    main()