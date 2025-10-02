import cv2
import numpy as np
def draw_player_stats(frames,player_stats):
    for i ,row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_shot_speed']
        player_1_average_player_speed = row['player_1_average_player_speed']
        player_2_average_player_speed = row['player_2_average_player_speed']
        player_1_average_shot_speed= row['player_1_average_shot_speed']
        player_2_average_shot_speed = row['player_2_average_shot_speed']
        frame=frames[i]
        shapes=np.zeros_like(frame,np.uint8)
        widths=350
        heights=230
        start_x=frame.shape[1]-400
        start_y=frame.shape[0]-500
        end_x=start_x+widths
        end_y=start_y+heights
        overlay=frame.copy()
        cv2.rectangle(overlay,(start_x,start_y),(end_x,end_y),(0,0,0),-1)
        alph=.5
        cv2.addWeighted(overlay,alph,frame,1-alph,0,frame)
        text="    Player 1      Player 2"
        frames[i]=cv2.putText(frames[i],text,(start_x+80,start_y+30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        text="shot speed"
        frames[i]=cv2.putText(frames[i],text,(start_x+10,start_y+80),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        text=f"{np.abs(player_1_shot_speed):.1f} km/h   {np.abs(player_2_shot_speed):.1f} km/h"
        frames[i]=cv2.putText(frames[i],text,(start_x+130,start_y+80),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
        text="player speed"
        frames[i]=cv2.putText(frames[i],text,(start_x+10,start_y+120),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        text=f"{np.abs(player_1_speed):.1f} km/h   {np.abs(player_2_speed):.1f} km/h"
        frames[i]=cv2.putText(frames[i],text,(start_x+130,start_y+120),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
        text="average. S .speed"
        frames[i]=cv2.putText(frames[i],text,(start_x+10,start_y+160),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        text=f"{np.abs(player_1_average_shot_speed):.1f} km/h   {np.abs(player_2_average_shot_speed):.1f} km/h"
        frames[i]=cv2.putText(frames[i],text,(start_x+130+20,start_y+160),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
        text="average. P .speed"
        frames[i]=cv2.putText(frames[i],text,(start_x+10,start_y+200),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        text=f"{np.abs(player_1_average_player_speed):.1f} km/h   {np.abs(player_2_average_player_speed):.1f} km/h"
        frames[i]=cv2.putText(frames[i],text,(start_x+130+20,start_y+200),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
    return frames