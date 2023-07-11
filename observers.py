import numpy as np
import supervision as sv

# MACROS
BH_THRES = 2

class observer_hd:
    # Class to handle all the layers of observers
    def __init__(self, team):
        # first layer
        self.players = {}
        self.basket = None
        self.ball = None
        self.active_bh = None
        # second layer
        self.observers_2 = {}
        # third layer
        self.teams = {0:team} #hardcoded for testing

    def upd_observers(self, detections):

        # Deactivate all players to filter
        self.players_deactivate()

        # Filter multiple balls
        detections = self.conf_filter(detections, 5)

        # Get the ball
        if detections[detections.class_id == 5]:
            if not self.ball:
                self.ball = ball_obs(detections[detections.class_id == 5])
            else: self.ball.upd_ball(detections[detections.class_id == 5])

        # Get the ballhandler
        if self.ball:
            # Assign bh to the player identified as bh that intersects with the ball

            potential_bh_tracker_id = {}
            # Potential bh intersects ball
            for xyxy, confidence, class_id, tracker_id in detections[detections.class_id == 0]:
                intersection = self.intersection(self.ball.xyxy, xyxy)
                if intersection > 0:
                    potential_bh_tracker_id[tracker_id] = class_id
            # list with bh that intersect with ball
            #detected_bh = list(potential_bh_tracker_id.keys())[list(potential_bh_tracker_id.values()).index(0)]
            detected_bh = list(potential_bh_tracker_id.keys())
            # select bh with higher confidence
            max_conf = 0
            for bh_tracker_id in detected_bh:
                if max_conf < detections[detections.tracker_id == bh_tracker_id].confidence:
                    max_conf = detections[detections.tracker_id == bh_tracker_id].confidence
                    self.active_bh = bh_tracker_id
            # filter rest of players
            detections = self.conf_filter_2(detections, class_id, self.active_bh)
        else:
            # if the ball is not detected the bh is the detected bh
            detections = self.conf_filter(detections, 0)
            if detections[detections.class_id == 0]:
                self.active_bh = detections[detections.class_id == 0].tracker_id
        # if a new bh is not detected, the active bh is the previous one

        # Update first and second layer
        for xyxy, confidence, class_id, tracker_id in detections:
            '''
            if (class_id == 0) or (class_id == 2):
                if tracker_id in self.players:
                    # right now this if else makes no sense but we'll see
                    self.players[tracker_id].upd_player(detections[detections.tracker_id == tracker_id])
                    print('player updated!')
                else: 
                    self.players[tracker_id] = player_obs(detections[detections.tracker_id == tracker_id])
                    print('player created!')
                
                if (class_id == 0):
                    if 'pass_obs' in self.observers_2:
                        self.observers_2['pass_obs'].upd_bh(self.players[tracker_id])
                        print('pass_obs updated')
                    else: 
                        self.observers_2['pass_obs'] = pass_obs(self.players[tracker_id])
                        print('pass_obs created')
            '''
            if class_id == 0:
                if tracker_id in self.players:
                    self.players[tracker_id].upd_player(detections[detections.tracker_id == tracker_id])
                    # keep player as class 2 until it gets to the threshold
                    #self.players[tracker_id].class_id = 2
                    self.players[tracker_id].bh_counter_inc()
                    #print('player updated!')
                else: 
                    self.players[tracker_id] = player_obs(detections[detections.tracker_id == tracker_id])
                    # keep player as class 2 until it gets to the threshold
                    #self.players[tracker_id].class_id = 2
                    self.players[tracker_id].bh_counter_inc()
                    #print('player created!')
            
            elif class_id == 2:
                
                if tracker_id in self.players:
                    self.players[tracker_id].upd_player(detections[detections.tracker_id == tracker_id])
                    self.players[tracker_id].bh_counter_reset()
                    #print('player updated!')
                else: 
                    self.players[tracker_id] = player_obs(detections[detections.tracker_id == tracker_id])
                    self.players[tracker_id].bh_counter_reset()
                    #print('player created!')

            elif class_id == 3 or class_id == 4:
                if not self.basket:
                    self.basket = basket_obs(detections[detections.class_id == class_id])
                else: 
                    self.basket.upd_basket(detections[detections.class_id == class_id])

        for player in iter(self.players.values()):
            # second layer
            if (player.class_id == 0):
                if 'pass_obs' in self.observers_2:
                    self.observers_2['pass_obs'].upd_bh(self.players[int(player.tracker_id)])
                    #print('pass_obs updated')
                else: 
                    print(player.tracker_id)
                    self.observers_2['pass_obs'] = pass_obs(self.players[int(player.tracker_id)])
                    #print('pass_obs created')

        # this could be made in another method
        for player in self.players:
            #player.upd_player_id(frame)
            pass
        
        # Update third layer
        self.teams[0].upd_players(self.players)

        for _, team in self.teams.items():
            team.upd_team_stats()

    def players_deactivate(self):
        for _, player in self.players.items():
            player.active = 0

    def conf_filter(self, detections, class_id):
        # mofify object detection as only one ball handler allowed
        filter_idx = np.where(detections.class_id == class_id)[0]
        #print(filter_idx)
        if filter_idx.size > 1:
            filtered_conf = detections.confidence[filter_idx]
            max_idx = filtered_conf.argmax(axis = 0)
            filter_idx = np.delete(filter_idx, max_idx)
            if class_id == 0:
                detections.class_id[filter_idx] = 2
            elif class_id == 5:
                detections.xyxy = np.delete(detections.xyxy, filter_idx)
                detections.class_id = np.delete(detections.class_id, filter_idx)
                detections.confidence = np.delete(detections.confidence, filter_idx)
                detections.tracker_id = np.delete(detections.tracker_id, filter_idx)
        #print(detections.class_id)
        return detections
    
    def conf_filter_2(self, detections, class_id, track_id):
        if track_id == None:
            pass
        else :
            filter_idx = np.where(detections.class_id == class_id)[0]
            track_id_idx = np.where(detections.tracker_id == track_id)[0]
            filter_idx = np.delete(filter_idx, track_id_idx)
            if class_id == 0:
                detections.class_id[filter_idx] = 2
        return detections

    def intersection(self, bb1, bb2):

        # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

        assert bb1[0] < bb1[2]
        assert bb1[1] < bb1[3]
        assert bb2[0] < bb2[2]
        assert bb2[1] < bb2[3]

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
        return intersection_area
    
    def export_obs(self):
        xyxy = np.empty((0,4))
        class_id = []
        confidence = np.array([])
        tracker_id = np.array([])
        if self.players:
            for player in iter(self.players.values()):
                if player._active:
                    xyxy = np.vstack([xyxy, player.xyxy])
                    class_id = np.append(class_id, int(player.class_id))
                    confidence = np.append(confidence, player.conf)
                    tracker_id = np.append(tracker_id, int(player.tracker_id))
                else: continue
            detections = sv.Detections(xyxy = xyxy, class_id = class_id.astype(int))
            detections.confidence = confidence
            detections.tracker_id = tracker_id
        else: detections = sv.Detections()
        return detections

    


### FIRST LAYER OF OBSERVERS ###
class player_obs:

    def __init__(self, detection):
        #self.xyxy, self.conf, self.class_id, self.tracker_id = detections[detections.tracker_id == tracker_id]
        self.xyxy = detection.xyxy[0]
        self.class_id = detection.class_id
        self.conf = detection.confidence
        self.tracker_id = detection.tracker_id

        self.frames = 1
        self._active = 1
        self.bh_count = 0

        self.team = None
        self._player_id = None

        #make getters and setters for all this things
        self.passes = 0
        self.fg = 0
        self.fg_made = 0
        self.rebounds = 0

    @property
    def player_id(self):
        return self._player_id
    
    @player_id.setter
    def player_id(self, new_id):
        self._player_id = new_id

    @property
    def active(self):
        return self._active
    
    @active.setter
    def active(self, state):
        if (state):
            self._active = 1
        else: self._active = 0

    def bh_counter_inc(self):
        self.bh_count += 1
        print(self.bh_count)
        print(BH_THRES)
        if self.bh_count >= BH_THRES:
            print('hey')
            self.class_id = 0
        else:
            self.class_id = 2 
        print(self.class_id)

    def bh_counter_reset(self):
        self.bh_count = 0
        self.class_id = 2

    def isbh(self):
        return (self.class_id == 0)

    def upd_player(self, detection):
        self.xyxy = detection.xyxy[0] 
        self.class_id = detection.class_id
        self.conf = detection.confidence
        self.tracker_id = detection.tracker_id

        self._active = 1
        self.frames += 1

    def upd_player_team(self, frame):
        if not self.team:
            # number identification function
            pass

    def upd_player_id(self, frame):
        if not self._player_id:
            # number identification function
            pass

class basket_obs:

    def __init__(self, detection):
        #self.xyxy, self.conf, self.class_id, self.tracker_id = detections[detections.tracker_id == tracker_id]
        self.xyxy = detection.xyxy[0] 
        self.class_id = detection.class_id 
        self.conf = detection.confidence
        self.frames = 1

    def upd_basket(self, detection):
        self.xyxy = detection.xyxy[0] 
        self.class_id = detection.class_id 
        self.conf = detection.confidence
        self.frames += 1

class ball_obs:

    def __init__(self, detection):
        self.xyxy = detection.xyxy[0]
        self.class_id = detection.class_id 
        self.conf = detection.confidence
        self.frames = 1

    def upd_ball(self, detection):
        self.xyxy = detection.xyxy[0] 
        self.class_id = detection.class_id 
        self.conf = detection.confidence
        self.frames += 1


### SECOND LAYER OF OBSERVERS ###

class pass_obs:
    def __init__(self, player_obs):
        self.current_bh = player_obs
        self.prev_bh = None
        self.pass_flag = 0

    def upd_bh(self, player):
        self.prev_bh = self.current_bh
        self.current_bh = player

        if self.prev_bh != None and self.prev_bh != self.current_bh:
            self.pass_flag = 1
            self.prev_bh.passes += 1
            print(f"pass from player {self.prev_bh} to player {self.current_bh}")
        else: self.pass_flag = 0


### THIRD LAYER OF OBSERVERS ###

class team:
    
    def __init__(self, team_id):
        self.team_id = team_id
        self.players = {}

        self.passes = 0
        self.fg = 0
        self.fg_made = 0
        self.rebounds = 0

    def upd_players(self, players):
        self.players = players

    def upd_team_stats(self):
        print('team stats updated')
        passes = 0
        fg = 0
        fg_made = 0
        rebounds = 0
        for _, player in self.players.items():
            passes += player.passes
            fg += player.fg
            fg_made += player.fg_made
            rebounds += player.rebounds

        self.passes = passes
        self.fg = fg
        self.fg_made = fg_made
        self.rebounds = rebounds

    def report(self):
        print(f"Passes made by team: {self.passes}")
    