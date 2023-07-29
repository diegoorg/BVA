import numpy as np
import supervision as sv
import easyocr
from utils import decision, intersection 

# MACROS
CLASSES = {'ball':0, 'ball-handler': 1, 'basket':2, 'made-basket':3, 'player':4}
BH_THRES = 2
MB_THRES = 2
TEAM_THRES = 10
ID_THRES = 0.9
W_MAX = [0.02370197, 0.01116409, -0.12604768]
TH = [-0.02705948498696889]


### OBSERVERS' HANDLER ###

class observer_hd:
    # Class to handle all the layers of observers
    def __init__(self, teams):
        # first layer
        self.players = {}
        self.basket = None
        self.ball = None
        self.active_bh = None
        self.old_bh = None
        # second layer
        self.observers_2 = {}
        # third layer
        self.teams = teams

        # Load OCR model
        self.reader = easyocr.Reader(['en'])
        self.reduced_class = '0123456789'
        

    def upd_observers(self, detections, frame):

        # Deactivate all players to filter
        self.players_deactivate()
        # Deactivate ball and basket to filter visualization
        self.ball_deactivate()
        self.basket_deactivate

        # Filter multiple balls, baskets and made-baskets
        detections = self.conf_filter(detections, CLASSES['ball'])
        detections = self.conf_filter(detections, CLASSES['basket'])
        detections = self.conf_filter(detections, CLASSES['made-basket'])

        # Get the ball
        if CLASSES['ball'] in detections.class_id:
            if not self.ball:
                self.ball = ball_obs(detections[detections.class_id == CLASSES['ball']])
            else: self.ball.upd_ball(detections[detections.class_id == CLASSES['ball']])
        else: self.ball = None

        # Get the ballhandler
        if not np.any(detections.class_id == CLASSES['ball-handler']):
            pass
        else:
            if self.ball:
                # Assign bh to the player identified as bh that intersects with the ball
                potential_bh_tracker_id = {}
                # Potential bh intersects ball
                for xyxy, confidence, class_id, tracker_id in detections[detections.class_id == CLASSES['ball-handler']]:
                    intersection = self.intersection(self.ball.xyxy, xyxy)
                    if intersection > 0:
                        potential_bh_tracker_id[tracker_id] = class_id
                # If there are no intersections, the detected bh are simply filtered
                if not potential_bh_tracker_id:
                    detections = self.conf_filter(detections, CLASSES['ball-handler'])
                    if detections[detections.class_id == CLASSES['ball-handler']]:
                        self.active_bh = detections.tracker_id[detections.class_id == CLASSES['ball-handler']]
                else:
                    # List with bh that intersect with ball
                    detected_bh = list(potential_bh_tracker_id.keys())
                    # Select bh with higher confidence
                    max_conf = 0
                    for bh_tracker_id in detected_bh:
                        if max_conf < detections.confidence[detections.tracker_id == bh_tracker_id]:
                            max_conf = detections.confidence[detections.tracker_id == bh_tracker_id]
                            self.active_bh = bh_tracker_id
                    # Filter rest of players
                    detections = self.conf_filter_2(detections, CLASSES['ball-handler'], self.active_bh)
            else:
                # If the ball is not detected the bh is the detected bh
                detections = self.conf_filter(detections, CLASSES['ball-handler'])
                self.active_bh = detections.tracker_id[detections.class_id == CLASSES['ball-handler']]

        # Update first and second layer
        self.active_bh = None
        for xyxy, confidence, class_id, tracker_id in detections:
            # Increment bh counter. Save info
            if class_id == CLASSES['ball-handler']:
                self.active_bh = tracker_id
                if tracker_id in self.players:
                    self.players[tracker_id].upd_player(detections[detections.tracker_id == tracker_id])
                    self.players[tracker_id].bh_counter_inc()
                else: 
                    self.players[tracker_id] = player_obs(detections[detections.tracker_id == tracker_id])
                    self.players[tracker_id].bh_counter_inc()
                if self.players[tracker_id].isbh():
                    self.old_bh = tracker_id
            # Reset bh cpunter. Save info
            elif class_id == CLASSES['player']:
                if tracker_id in self.players:
                    self.players[tracker_id].upd_player(detections[detections.tracker_id == tracker_id])
                    self.players[tracker_id].bh_counter_reset()
                else: 
                    self.players[tracker_id] = player_obs(detections[detections.tracker_id == tracker_id])
                    self.players[tracker_id].bh_counter_reset()
            # Save basket info
            elif class_id == CLASSES['basket'] or class_id == CLASSES['made-basket']:
                if not self.basket:
                    self.basket = basket_obs(detections[detections.class_id == class_id])
                else: 
                    self.basket.upd_basket(detections[detections.class_id == class_id])        
        
        # Keep old bh if there is none
        if self.active_bh == None:
            if self.old_bh != None:
                self.players[self.old_bh].class_id = CLASSES['ball-handler']
        else:
            if self.players[self.active_bh].isbh():
                pass
            elif self.old_bh != None:
                self.players[self.old_bh].class_id = CLASSES['ball-handler']

        # Update second layer and execute player Identification
        for player in iter(self.players.values()):
            if player.active:
                # second layer
                if (player.class_id == CLASSES['ball-handler']):
                    if 'pass_obs' in self.observers_2:
                        self.observers_2['pass_obs'].upd_bh(self.players[int(player.tracker_id)])
                    else: 
                        print(player.tracker_id)
                        self.observers_2['pass_obs'] = pass_obs(self.players[int(player.tracker_id)])
                    if 'fg_obs' in self.observers_2:
                        self.observers_2['fg_obs'].upd_fg(self.players[int(player.tracker_id)])
                    else: 
                        self.observers_2['fg_obs'] = fg_obs(self.players[int(player.tracker_id)])
                # Player ID
                player.upd_player_id(frame, self.reader, self.reduced_class)

        # Update field goal information if there was no bh
        if len([player for player in iter(self.players.values()) if player.class_id == CLASSES['ball-handler']]) == 0:
            if 'fg_obs' in self.observers_2:
                self.observers_2['fg_obs'].upd_fg(None)
            else: 
                self.observers_2['fg_obs'] = fg_obs(None)
        
        # Trigger FGA/FGM
        if self.ball and self.basket:
            self.observers_2['fg_obs'].check_posible_fg(self.ball, self.basket)
        if detections[detections.class_id == CLASSES['made-basket']]:
            self.observers_2['fg_obs'].mb_counter_inc()
        else: self.observers_2['fg_obs'].mb_counter_reset()

        # Update third layer
        for _, team in self.teams.items():
            team.upd_players(self.players)
            team.upd_team_stats()
            team.upd_player_stats()

    def players_deactivate(self):
        for _, player in self.players.items():
            player.active = 0
    
    def ball_deactivate(self):
        if self.ball != None:
            self.ball.active = 0

    def basket_deactivate(self):
        if self.basket != None:
            self.basket.active = 0

    def conf_filter(self, detections, class_id):
        # Mofify object detection as only one ball handler allowed
        # Delete or modify detections to keep higher confidence object
        filter_idx = np.where(detections.class_id == class_id)[0]
        if filter_idx.size > 1:
            filtered_conf = detections.confidence[filter_idx]
            max_idx = filtered_conf.argmax(axis = 0)
            filter_idx = np.delete(filter_idx, max_idx)
            if class_id == CLASSES['ball-handler']:
                # Change bh to player
                detections.class_id[filter_idx] = CLASSES['player']
            else:
                # Delete lower confidence detections
                detections.xyxy = np.delete(detections.xyxy, filter_idx)
                detections.class_id = np.delete(detections.class_id, filter_idx)
                detections.confidence = np.delete(detections.confidence, filter_idx)
                detections.tracker_id = np.delete(detections.tracker_id, filter_idx)
        return detections
    
    def conf_filter_2(self, detections, class_id, track_id):
        # Keep specific object as the only object for class_id class
        if track_id == None:
            pass
        else :
            filter_idx = np.where(detections.class_id == class_id)[0]
            track_id_idx = np.where(detections.tracker_id == track_id)[0]
            filter_idx = np.delete(filter_idx, np.where(filter_idx == track_id_idx)[0])
            if class_id == CLASSES['ball-handler']:
                detections.class_id[filter_idx] = CLASSES['player']
        return detections

    def intersection(self, bb1, bb2):
        # Intersection measurement between 2 given bbox
        '''
        Code obtained from:
        https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
        '''
        # Check input data
        assert bb1[0] < bb1[2]
        assert bb1[1] < bb1[3]
        assert bb2[0] < bb2[2]
        assert bb2[1] < bb2[3]

        # Determine the coordinates of the intersection rectangle
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
        # Export observers for annotation
        # The output is a sv.Detections object
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
        if self.basket:
            if self.basket.active:
                xyxy = np.vstack([xyxy, self.basket.xyxy])
                class_id = np.append(class_id, int(self.basket.class_id))
                confidence = np.append(confidence, self.basket.conf)
                tracker_id = np.append(tracker_id, int(self.basket.tracker_id))
        if self.ball:
            if self.ball.active:
                xyxy = np.vstack([xyxy, self.ball.xyxy])
                class_id = np.append(class_id, int(self.ball.class_id))
                confidence = np.append(confidence, self.ball.conf)
                tracker_id = np.append(tracker_id, int(self.ball.tracker_id))

        detections = sv.Detections(xyxy = xyxy, class_id = class_id.astype(int))
        detections.confidence = confidence
        detections.tracker_id = tracker_id
        return detections
    
    def export_ply_id(self):
        # Export player ID as a dict
        player_id = {}
        for player in iter(self.players.values()):
            player_id[player.tracker_id[0]] = player.player_id
        return player_id
    
    def export_ply_team(self):
        # Export player team as a dict
        player_team = {}
        for player in iter(self.players.values()):
            player_team[player.tracker_id[0]] = player.team
        return player_team

    
### FIRST LAYER OF OBSERVERS ###

class player_obs:

    def __init__(self, detection):
        # Player basic info
        self.xyxy = detection.xyxy[0]
        self.class_id = detection.class_id
        self.conf = detection.confidence
        self.tracker_id = detection.tracker_id
        # BH info
        self.frames = 1
        self._active = 1
        self.bh_count = 0
        # Team and ID info
        self.team = None
        self.team_tmp = None
        self.team_cnt = 0
        self._player_id = None
        # Stats
        self._passes = 0
        self._fga = 0
        self._fgm = 0
        self._reb = 0

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

    @property
    def passes(self):
        return self._passes
    
    @passes.setter
    def passes(self, p):
        self._passes = p

    @property
    def fga(self):
        return self._fga
    
    @fga.setter
    def fga(self, p):
        self._fga = p

    @property
    def fgm(self):
        return self._fgm
    
    @fgm.setter
    def fgm(self, p):
        self._fgm = p

    @property
    def reb(self):
        return self._reb
    
    @reb.setter
    def reb(self, p):
        self._reb = p

    def bh_counter_inc(self):
        # Considere player as bh after several (BH_THRES) detections
        self.bh_count += 1
        if self.bh_count >= BH_THRES:
            self.class_id = CLASSES['ball-handler']
        else:
            self.class_id = CLASSES['player'] 

    def bh_counter_reset(self):
        self.bh_count = 0
        self.class_id = CLASSES['player']

    def isbh(self):
        # Check if object is bh
        return (self.class_id == CLASSES['ball-handler'])

    def upd_player(self, detection):
        # Update object basic information with detection
        self.xyxy = detection.xyxy[0] 
        self.class_id = detection.class_id
        self.conf = detection.confidence
        self.tracker_id = detection.tracker_id

        self._active = 1
        self.frames += 1

    def upd_player_id(self, frame, reader, reduced_class):
        # Function for team classification and number recognition
        if not self.player_id or self.player_id == 'unk' or self.team_cnt < TEAM_THRES:
            # Number identification function
            frame_h, frame_w, _ = frame.shape

            # All numbers positive integers
            list1 = np.asarray(self.xyxy, dtype = 'int')
            x1,y1,x2,y2 = [(i > 0) * i for i in list1]

            # Hight and width
            h = y2-y1
            w = x2-x1

            # Make sure all positive
            a = int(max(0, y1+0.25*h))
            b = int(min(frame_h, y2-0.5*h))
            c = int(max(0, x1+0.25*w))
            d = int(min(frame_w, x2-0.25*w))

            # Crop original frame
            crop = frame[a:b,c:d]

            # Execute team classification
            if self.team_cnt < TEAM_THRES:
                tmp = decision(crop, W_MAX, TH)
                if self.team_tmp == tmp:
                    self.team_cnt += 1
                else: self.team_cnt = 0
                self.team_tmp = tmp
                if self.team_cnt == TEAM_THRES:
                    self.team = tmp
            # Execute number recognition
            if not self.player_id or self.player_id == 'unk':
                result = reader.readtext(crop, allowlist = reduced_class)
                if result != []:
                    _, id_num, id_conf = result[0]
                    # Only consider numbers with confidence higher than ID_THRES
                    if id_conf >= ID_THRES:
                        self.player_id = id_num
                    else: self.player_id = 'unk'
                else: self.player_id = 'unk'

class basket_obs:

    def __init__(self, detection):
        # Basket basic info
        self.xyxy = detection.xyxy[0] 
        self.class_id = detection.class_id 
        self.conf = detection.confidence
        self.tracker_id = detection.tracker_id
        self.frames = 1
        self.active = 1

    def upd_basket(self, detection):
        # Update basic info
        self.xyxy = detection.xyxy[0] 
        self.class_id = detection.class_id 
        self.conf = detection.confidence
        self.tracker_id = detection.tracker_id
        self.frames += 1
        self.active = 1

class ball_obs:

    def __init__(self, detection):
        # Ball basic info
        self.xyxy = detection.xyxy[0]
        self.class_id = detection.class_id 
        self.conf = detection.confidence
        self.tracker_id = detection.tracker_id
        self.frames = 1
        self.active = 1

    def upd_ball(self, detection):
        # Update basic info
        self.xyxy = detection.xyxy[0] 
        self.class_id = detection.class_id 
        self.conf = detection.confidence
        self.tracker_id = detection.tracker_id
        self.frames += 1
        self.active = 1


### SECOND LAYER OF OBSERVERS ###

class pass_obs:
    def __init__(self, player_obs):
        # Pass basic info
        self.current_bh = player_obs
        self.prev_bh = None
        self.pass_flag = 0

    def upd_bh(self, player):
        # Update bh info
        self.prev_bh = self.current_bh
        self.current_bh = player
        # When the bh changes in the same team, pass assigned to previous bh
        if self.prev_bh != None and self.prev_bh != self.current_bh and self.prev_bh.team == self.current_bh.team:
            self.pass_flag = 1
            self.prev_bh.passes += 1
            print(f"pass from player {self.prev_bh.tracker_id} to player {self.current_bh.tracker_id}")
        else: self.pass_flag = 0

class fg_obs:

    def __init__(self,player):
        # FG basic info
        self.current_bh = player
        self.fgm_flag = 0
        self.reb_flag = 0
        # Made-basket counter
        self.mb_count = 0
        # FGA and FGM timers
        self.fga_timer = 0
        self.fgm_timer = 0

    def upd_fg(self, player):
        # Update basic info
        self.current_bh = player
        # Decrement timer for FGA
        if self.fga_timer:
            self.fga_timer -= 1
        # Assign rebound
        elif player != None and self.reb_flag:
            self.reb_flag = 0
            self.current_bh.reb = self.current_bh.reb + 1
        # Decrement timer for FGM
        if self.fgm_timer:
            self.fgm_timer -= 1
        # Assign made-basket to bh
        if self.fgm_flag and self.current_bh and (player != None):
            self.current_bh.fgm = self.current_bh.fgm + 1
            print(f"field goal made by player {self.current_bh.tracker_id}")
            self.fgm_flag = 0

    def check_posible_fg(self, ball, basket):
        # Consider a FGA when ball and basket intersect
        if self.fga_timer:
            pass
        else: 
            if intersection(ball.xyxy, basket.xyxy):
                self.reb_flag = 1
                self.fga_timer = 90 #frame cooldown: 3 seconds (3*30fps)
                self.current_bh.fga = self.current_bh.fga + 1
                print(f"field goal attempt from player {self.current_bh.tracker_id}")
    
    def mb_counter_inc(self):
        # Increment the made-basket counter
        self.mb_count += 1
        if self.mb_count == MB_THRES and not(self.fgm_timer):
            self.fgm_timer = 60 #frame cooldown: 2 seconds (2*30fps) 
            self.fgm_flag = 1

    def mb_counter_reset(self):
        # Reset made-basket counter
        self.mb_count = 0


### Stats Container ###

class team:
    
    def __init__(self, team_id):
        # Team basic info
        self.team_id = team_id
        self.players = {}
        self.identified_players = {}
        # Stats
        self.passes = 0
        self.fga = 0
        self.fgm = 0
        self.reb = 0
        self.player_stats = {}

    def upd_players(self, players):
        # Save players that belong to the team
        # Save identified player for individual stats
        for player in iter(players.values()):
            if player.team == self.team_id:
                self.players[player.tracker_id[0]] = player
                if player.player_id != None and player.player_id != 'unk':
                    self.identified_players[player.tracker_id[0]] = player

    def upd_team_stats(self):
        # Gather team stats
        print(f"Team {self.team_id} stats updated")
        passes = 0
        fg = 0
        fg_made = 0
        rebounds = 0
        for _, player in self.players.items():
            passes += player.passes
            fg += player.fga
            fg_made += player.fgm
            rebounds += player.reb

        self.passes = passes
        self.fga = fg
        self.fgm = fg_made
        self.reb = rebounds

    def upd_player_stats(self):
        # Gather individual stats
        print(f"Team {self.team_id} player stats updated")
        player_stats = {}
        for player in iter(self.identified_players.values()):
            if player.player_id in player_stats:
                dict = self.player_stats[player.player_id]
                player_stats[player.player_id] = {'passes': dict['passes'] + player.passes,
                                                        'fga': dict['fga'] + player.fga,
                                                        'fgm': dict['fgm'] + player.fgm,
                                                        'reb': dict['reb'] + player.reb}
            else: 
                player_stats[player.player_id] = {'passes': player.passes,
                                                        'fga': player.fga,
                                                        'fgm': player.fgm,
                                                        'reb': player.reb}
        self.player_stats = player_stats

    def report(self):
        # Report team and individual stats
        print("************")
        print(f"Team {self.team_id} stats:")
        print(f"Passes: {self.passes}")
        print(f"FGA: {self.fga}")
        print(f"FGM: {self.fgm}")
        fg_per = 0
        if self.fga != 0:
            fg_per = self.fgm/self.fga*100
        print(f"FG%: {fg_per}")
        print(f"Rebounds: {self.reb}")
        print("------------")
        for id, player in self.player_stats.items():
            print(f"Player {id} stats:")
            print(f"Passes: {player['passes']}")
            print(f"FGA: {player['fga']}")
            print(f"FGM: {player['fgm']}")
            fg_per = 0
            if player['fga'] != 0:
                fg_per = player['fgm']/player['fga']*100
            print(f"FG%: {fg_per}")
            print(f"Rebounds: {player['reb']}")
            print("------------")
        print("************")   