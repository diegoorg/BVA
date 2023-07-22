import numpy as np
import supervision as sv
import easyocr
from utils import decision, intersection 

# MACROS
CLASSES = {'ball':0, 'ball-handler': 1, 'basket':2, 'made-basket':3, 'player':4}
BH_THRES = 5
TEAM_THRES = 5
ID_THRES = 0.9
MB_THRES = 2
TH = [0.4934562550056104, 0.44458746597784793, 0.4529725222840385]

class observer_hd:
    # Class to handle all the layers of observers
    def __init__(self, team):
        # first layer
        self.players = {}
        self.basket = None
        self.ball = None
        self.active_bh = None
        self.old_bh = None

        # second layer
        self.observers_2 = {}

        # third layer
        self.teams = {0:team} #hardcoded for testing

        # Load OCR model
        self.reader = easyocr.Reader(['en'])
        self.reduced_class = '0123456789'
        

    def upd_observers(self, detections, frame):

        # Deactivate all players to filter
        self.players_deactivate()

        # Filter multiple balls, baskets and made-baskets
        detections = self.conf_filter(detections, CLASSES['ball'])
        detections = self.conf_filter(detections, CLASSES['basket'])
        detections = self.conf_filter(detections, CLASSES['made-basket'])

        # Get the ball
        if detections[detections.class_id == CLASSES['ball']]:
            if not self.ball:
                self.ball = ball_obs(detections[detections.class_id == CLASSES['ball']])
            else: self.ball.upd_ball(detections[detections.class_id == CLASSES['ball']])
        else: self.ball = None

        # Get the ballhandler
            # if a new bh is not detected, the active bh is the previous one
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
                    # list with bh that intersect with ball
                    #detected_bh = list(potential_bh_tracker_id.keys())[list(potential_bh_tracker_id.values()).index(0)]
                    detected_bh = list(potential_bh_tracker_id.keys())
                    # select bh with higher confidence
                    max_conf = 0
                    for bh_tracker_id in detected_bh:
                        if max_conf < detections.confidence[detections.tracker_id == bh_tracker_id]:
                            max_conf = detections.confidence[detections.tracker_id == bh_tracker_id]
                            self.active_bh = bh_tracker_id
                    # filter rest of players
                    detections = self.conf_filter_2(detections, CLASSES['ball-handler'], self.active_bh)
            else:
                # if the ball is not detected the bh is the detected bh
                detections = self.conf_filter(detections, CLASSES['ball-handler'])
                #if detections[detections.class_id == CLASSES['ball-handler']]:
                self.active_bh = detections.tracker_id[detections.class_id == CLASSES['ball-handler']]
        detections.class_id[detections.tracker_id == self.active_bh] = CLASSES['ball-handler']    

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
            if class_id == CLASSES['ball-handler']:
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
                if not self.players[tracker_id].isbh():
                    self.active_bh = self.old_bh
            
            elif class_id == CLASSES['player']:
                if tracker_id in self.players:
                    self.players[tracker_id].upd_player(detections[detections.tracker_id == tracker_id])
                    self.players[tracker_id].bh_counter_reset()
                    #print('player updated!')
                else: 
                    self.players[tracker_id] = player_obs(detections[detections.tracker_id == tracker_id])
                    self.players[tracker_id].bh_counter_reset()
                    #print('player created!')

            elif class_id == CLASSES['basket'] or class_id == CLASSES['made-basket']:
                if not self.basket:
                    self.basket = basket_obs(detections[detections.class_id == class_id])
                else: 
                    self.basket.upd_basket(detections[detections.class_id == class_id])        
        
        # Keep old bh if there is none
        if (self.active_bh == self.old_bh):
            detections.class_id[detections.tracker_id == self.active_bh] = CLASSES['ball-handler']

        for player in iter(self.players.values()):
            if player.active:
                # second layer
                if (player.class_id == CLASSES['ball-handler']):
                    if 'pass_obs' in self.observers_2:
                        self.observers_2['pass_obs'].upd_bh(self.players[int(player.tracker_id)])
                        #print('pass_obs updated')
                    else: 
                        print(player.tracker_id)
                        self.observers_2['pass_obs'] = pass_obs(self.players[int(player.tracker_id)])
                        #print('pass_obs created')
                    if 'fg_obs' in self.observers_2:
                        self.observers_2['fg_obs'].upd_fg(self.players[int(player.tracker_id)])
                        #print('pass_obs updated')
                    else: 
                        print(player.tracker_id)
                        self.observers_2['fg_obs'] = fg_obs(self.players[int(player.tracker_id)])
                        #print('pass_obs created')
                # Player ID
                player.upd_player_id(frame, self.reader, self.reduced_class)

        # Update field goal information
        # If there was no bh
        if len([player for player in iter(self.players.values()) if player.class_id == CLASSES['ball-handler']]) == 0:
            if 'fg_obs' in self.observers_2:
                self.observers_2['fg_obs'].upd_fg(None)
                #print('pass_obs updated')
            else: 
                print(player.tracker_id)
                self.observers_2['fg_obs'] = fg_obs(None)
                #print('pass_obs created')
        if self.ball and self.basket:
            self.observers_2['fg_obs'].check_posible_fg(self.ball, self.basket)
        if detections[detections.class_id == CLASSES['made-basket']]:
            self.observers_2['fg_obs'].mb_counter_inc()
        else: self.observers_2['fg_obs'].mb_counter_reset()

        
        # Update third layer
        self.teams[0].upd_players(self.players)

        for _, team in self.teams.items():
            team.upd_team_stats()
        
        # Update flags
        self.old_bh = self.active_bh

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
            if class_id == CLASSES['ball-handler']:
                detections.class_id[filter_idx] = CLASSES['player']
            else:  #class_id == CLASSES['ball']:
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
            #print(filter_idx)
            track_id_idx = np.where(detections.tracker_id == track_id)[0]
            #print(track_id_idx)
            filter_idx = np.delete(filter_idx, np.where(filter_idx == track_id_idx)[0])
            #print(filter_idx)
            if class_id == CLASSES['ball-handler']:
                detections.class_id[filter_idx] = CLASSES['player']
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
    
    def export_ply_id(self):
        player_id = {}
        for player in iter(self.players.values()):
            player_id[player.tracker_id[0]] = player.player_id
        return player_id
    
    def export_ply_team(self):
        player_team = {}
        for player in iter(self.players.values()):
            player_team[player.tracker_id[0]] = player.team
        return player_team

    


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
        self.team_tmp = None
        self.team_cnt = 0
        self._player_id = None

        #make getters and setters for all this things
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
        self.bh_count += 1
        if self.bh_count >= BH_THRES:
            self.class_id = CLASSES['ball-handler']
        else:
            self.class_id = CLASSES['player'] 

    def bh_counter_reset(self):
        self.bh_count = 0
        self.class_id = CLASSES['player']

    def isbh(self):
        return (self.class_id == CLASSES['ball-handler'])

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

    def upd_player_id(self, frame, reader, reduced_class):
        if not self.player_id or self.player_id == 'unk' or self.team_cnt < TEAM_THRES:
            # number identification function
            frame_h, frame_w, _ = frame.shape

            # all numbers positive integers
            list1 = np.asarray(self.xyxy, dtype = 'int')
            x1,y1,x2,y2 = [(i > 0) * i for i in list1]
            #print(x1,x2,y1,y2)

            # hight and width
            h = y2-y1
            w = x2-x1

            # make sure all positive
            a = int(max(0, y1+0.25*h))
            b = int(min(frame_h, y2-0.5*h))
            c = int(max(0, x1+0.25*w))
            d = int(min(frame_w, x2-0.25*w))

            crop = frame[a:b,c:d]

            #img = Image.fromarray(crop, 'RGB')
            #img.save(f"test{index}.jpeg")
            #index += 1
            if self.team_cnt < TEAM_THRES:
                tmp = decision(TH, crop)
                if self.team_tmp == tmp:
                    self.team_cnt += 1
                else: self.team_cnt = 0
                self.team_tmp = tmp
                if self.team_cnt == TEAM_THRES:
                    self.team = tmp
            if not self.player_id or self.player_id == 'unk':
                result = reader.readtext(crop, allowlist = reduced_class)
                #print(result)
                if result != []:
                    id_box, id_num, id_conf = result[0]
                    if id_conf >= ID_THRES:
                        self.player_id = id_num
                    else: self.player_id = 'unk'
                else: self.player_id = 'unk'


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

class fg_obs:
    def __init__(self,player):
        self.current_bh = player
        self.fgm_flag = 0
        self.reb_flag = 0
        self.mb_count = 0
        self.fga_timer = 0
        self.fgm_timer = 0

    def upd_fg(self, player):
        self.current_bh = player
        if self.fga_timer:
            self.fga_timer -= 1
        elif player != None and self.reb_flag:
            self.reb_flag = 0
            self.current_bh.reb = self.current_bh.reb + 1
        if self.fgm_timer:
            self.fgm_timer -= 1
        if self.fgm_flag and self.current_bh and (player != None):
            self.current_bh.fgm = self.current_bh.fgm + 1
            print(f"field goal made by player {self.current_bh}")
            self.fgm_flag = 0

    def check_posible_fg(self, ball, basket):
        if self.fga_timer:
            pass
        else: 
            if intersection(ball.xyxy, basket.xyxy):
                self.reb_flag = 1
                self.fga_timer = 90 #frame cooldown: 3 seconds (3*30fps)
                self.current_bh.fga = self.current_bh.fga + 1
                print(f"field goal attempt from player {self.current_bh}")
    
    def mb_counter_inc(self):
        self.mb_count += 1
        if self.mb_count == MB_THRES and not(self.fgm_timer):
            self.fgm_timer = 60 #frame cooldown: 2 seconds (2*30fps) 
            self.fgm_flag = 1

    def mb_counter_reset(self):
        self.mb_count = 0


### THIRD LAYER OF OBSERVERS ###

class team:
    
    def __init__(self, team_id):
        self.team_id = team_id
        self.players = {}

        self.passes = 0
        self.fga = 0
        self.fgm = 0
        self.reb = 0

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
            fg += player.fga
            fg_made += player.fgm
            rebounds += player.reb

        self.passes = passes
        self.fga = fg
        self.fgm = fg_made
        self.reb = rebounds

    def report(self):
        print(f"Passes made by team: {self.passes}")
        print(f"FGA made by team: {self.fga}")
        print(f"FGM made by team: {self.fgm}")
        print(f"Rebounds made by team: {self.reb}")
    