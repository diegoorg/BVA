import supervision as sv
import numpy as np
from observers import observer_hd, team

 
xyxy = np.array([[     1565.7,      497.52,      1663.4,      738.07],
                [     1296.7,      520.16,      1418.2,      809.67],
                [     714.68,      377.88,      803.95,      619.74],
                [     814.94,      484.88,      961.66,      738.84],
                [       1299,      390.85,      1400.9,      688.23],
                [     547.41,      410.36,      634.05,         634],
                [     479.46,      335.73,      564.15,      509.07],
                [     815.75,      293.38,      917.52,      499.89],
                [     1580.7,      600.52,      1590.4,      610.07]])



class_id = np.array([0, 0, 2, 2, 2, 2, 2, 2, 5])
detections = sv.Detections(xyxy = xyxy, class_id = class_id)
detections.confidence = np.array([     0.8518,     0.51069,     0.73517,     0.82727,     0.68744,     0.81405,      0.8198,     0.73094,     0.73094])
detections.tracker_id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

team_0 = team(0)
observer = observer_hd(team_0)

# frame 1
observer.upd_observers(detections)
print(player.player_id for player in observer.players)

print(observer.players)

#frame 2
detections.class_id = np.array([2, 0, 2, 2, 2, 2, 2, 2, 5])
observer.upd_observers(detections)
print(player.player_id for player in observer.players)

print(observer.players)
print(team_0.report())