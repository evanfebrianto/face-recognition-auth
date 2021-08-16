import cv2
import sys
import pygame
import argparse
import numpy as np
from pygame.locals import *
from src import FaceRecognition
import mediapipe as mp
import os
import csv
from datetime import datetime
from reset import Reset

os.environ["GLOG_minloglevel"] ="3"
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0, help="videoSource")
    args = vars(ap.parse_args())

    face_recognition = FaceRecognition()
    game = GameScreen()
    game._init_main_screen()
    game._init_game_screen()
    game._init_inference_screen(args, face_recognition)
    game._init_inventory_screen()

    while True:
        game._on_loop()
        game._on_render()

        pygame.display.update()
        game._on_detect_input()

class GameScreen:
    def __init__(self):
        # screen initialization
        pygame.init()
        pygame.display.set_caption("Face Recognition")
        self._init_screen()
        
    def _init_screen(self):
        self.screen_idx = 0 #default show the init screen
        self.screen = pygame.display.set_mode([1350,650])
        self.mouse = None
        self.reset_logs_trigger = False
        self.reset = Reset()
    

    def _init_game_screen(self):
        # game screen initialization
        self.map_locked = pygame.image.load('./static/map_locked.jpg')
        self.map_unlocked = pygame.image.load('./static/map_unlocked.jpg')
        self.char_idle = pygame.image.load('./static/sprite/idle.png')
        self.char_run = pygame.image.load('./static/sprite/run.png')
        self.char_cur_position = [990, 320]
        self.char_tar_position = [990, 320]
        self.char_move_bool = [False, False, False, False]
        
        self.mask_locked = cv2.imread('./static/mask_locked.jpg')
        self.mask_locked = cv2.cvtColor(self.mask_locked, cv2.COLOR_BGR2GRAY)
        self.mask_locked[self.mask_locked >= 1] = 1
        self.mask_locked[self.mask_locked < 1] = 0
        
        self.mask_unlocked = cv2.imread('./static/mask_unlocked.jpg')
        self.mask_unlocked = cv2.cvtColor(self.mask_unlocked, cv2.COLOR_BGR2GRAY)
        self.mask_unlocked[self.mask_unlocked >= 1] = 1
        self.mask_unlocked[self.mask_unlocked < 1] = 0

        self.map_status = False
        self.map_status_last = False
        self.active_map = self.map_locked
        self.active_mask = self.mask_locked

        # find trigger
        self.trigger_in = [881, 409, 925, 423]
        self.trigger_out = [881, 424, 925, 440]
        self.trigger_stat_bool = -1
        self.trigger_stat_bool_last = -1
        self.trigger_inventory = [593, 521, 634, 566]
        self.trigger_inventory_stat = False

        # arrow label
        self.arrow_image = pygame.image.load('./static/arrow.png')
        self.arrow_image = pygame.transform.scale(self.arrow_image, (45, 30))
        self.arrow_image = pygame.transform.rotate(self.arrow_image, -90)

        # trigger font
        self.trigger_font = pygame.font.SysFont('constantia', 20)


    def _init_main_screen(self):
        self.menu_screen = pygame.image.load('./static/page_main.png')
        self.inference_screen = pygame.image.load('./static/page_inference.png')
        
        # button configuration
        self.color_white = (255, 255, 255)
        self.color_green = (0, 255, 0)
        self.color_blue = (0, 0, 128)
        self.color_red = (72, 0, 50)
        
        # light and dark shade of the button
        self.color_light = (170,170,170)  
        self.color_dark = (100,100,100)
        
        # screen and button size
        self.width = self.screen.get_width()
        self.height = 280
        self.button_w, self.button_h = 210, 45
        self.distance_between_buttons = 70
        self.button_status = None
        self.temp_button_status = None

    def _init_inference_screen(self, args, facerecognition):
        self.camera = cv2.VideoCapture(args['source'])
        self.font = pygame.font.SysFont('constantia', 40)
        self.face_recognition = facerecognition
        self.face_output = None
        
    def _init_inventory_screen(self):
        def load_csv_init(path='./report.csv'):
            csv_time, csv_pic, csv_logs = [], [], []
            with open(os.path.join(path)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for i, row in enumerate(csv_reader):
                    if(i > 0):
                        csv_time.append(row[0])
                        csv_pic.append(row[1])
                        csv_logs.append(row[2])
            output = [str(a + ", " + b + ", " + c) for a, b, c in zip(csv_time, csv_pic, csv_logs)]
            return output

        self.inventory_screen = pygame.image.load('./static/page_inventory.png')
        self.font_inventory = pygame.font.SysFont('constantia', 30)
        self.inventory_name = self.reset.inventory_name
        self.inventory_count = self.reset.inventory_current_count
        self.inventory_count_temp = self.inventory_count.copy()
        self.inventory_add_button = np.zeros(len(self.inventory_count), dtype=int)
        self.inventory_sub_button = np.zeros(len(self.inventory_count), dtype=int)
        self.inventory_change_trigger = False
        self.inventory_save_trigger = False
        self.inventory_count_min = 0
        self.inventory_count_max = 20
        self.inventory_count_diff = [0 for i in range(len(self.inventory_count))]
        
        self.inventory_logs_max = 16
        self.inventory_font_history = pygame.font.SysFont('constantia', 15)
        self.inventory_logs = load_csv_init()
        self.inventory_logs.append(None)
        self.inventory_logs = self.inventory_logs[-self.inventory_logs_max:]

        self.button_font_addsub = pygame.font.SysFont('constantia', 25)
        self.button_color1_addsub = (255, 245, 240)
        self.button_color2_addsub = (165, 42, 0)
        self.personal_belongings = [None for i in self.inventory_name] #by default


    def _on_loop(self):
        self.mouse = pygame.mouse.get_pos()
        if(self.screen_idx == 0):
            pass
        elif(self.screen_idx == 1):
            self.update_game_params()
        elif(self.screen_idx == 2):
            # self.update_inference_params()
            pass
        elif(self.screen_idx == 3):
            self.update_inventory_params()


    def _on_render(self):
        if(self.screen_idx == 0):
            self.update_main_screen()
        elif(self.screen_idx == 1):
            self.update_game_screen()
        elif(self.screen_idx == 2):
            self.update_inference_screen()
        elif(self.screen_idx == 3):
            self.update_inventory_screen()
        
        

    def _on_detect_input(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

             #checks if a mouse is clicked
            if ev.type == pygame.MOUSEBUTTONDOWN:
                # for screen menu case
                if(self.screen_idx == 0):
                    if(self.button_status[0] == "1"):
                        self.screen_idx = 1
                    if(self.button_status[1] == "1"):
                        _ = self.face_recognition.create_landmark_dict()
                        self.reset.reset_csv_logs()
                        self.reset._initialize_inventory()
                        self.inventory_count = self.reset.inventory_count.copy()
                        self.reset_logs_trigger = True
                        print("New identities generated")
                        print('Logs Reset')
                    if(self.button_status[2] == "1"):
                        self.reset.reset_csv_logs()
                        self.reset._initialize_inventory()
                        self.inventory_count = self.reset.inventory_count.copy()
                        self.reset_logs_trigger = True
                        print('Logs Reset')
                elif(self.screen_idx == 3):
                    self.inventory_change_trigger = True


            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_LEFT:
                    self.char_move_bool[0] = True
                if ev.key == pygame.K_RIGHT:
                    self.char_move_bool[1] = True
                if ev.key == pygame.K_UP:
                    self.char_move_bool[2] = True
                if ev.key == pygame.K_DOWN:
                    self.char_move_bool[3] = True
                if ev.key == pygame.K_q:
                    if(self.screen_idx == 1):
                        self.screen_idx = 0
                    elif(self.screen_idx == 2):
                        self.screen_idx = 1
                    elif(self.screen_idx == 3):
                        self.inventory_save_trigger = True
                if ev.key == pygame.K_RETURN:
                    if(self.screen_idx == 3):
                        self.inventory_save_trigger = True
                    elif(self.screen_idx == 2):
                        self.screen_idx = 1
                    elif(self.screen_idx == 1):
                        if(self.trigger_inventory_stat):
                            self.screen_idx = 3
                        elif(self.trigger_stat_bool==0):
                            self.screen_idx = 2
                    
            if ev.type == pygame.KEYUP:
                if ev.key == pygame.K_LEFT:
                    self.char_move_bool[0] = False
                if ev.key == pygame.K_RIGHT:
                    self.char_move_bool[1] = False
                if ev.key == pygame.K_UP:
                    self.char_move_bool[2] = False
                if ev.key == pygame.K_DOWN:
                    self.char_move_bool[3] = False

    def load_csv(self, path='./report.csv'):
        csv_time, csv_pic, csv_logs = [], [], []
        with open(os.path.join(path)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if(i > 0):
                    csv_time.append(row[0])
                    csv_pic.append(row[1])
                    csv_logs.append(row[2])
        self.inventory_logs = [str(a + ", " + b + ", " + c) for a, b, c in zip(csv_time, csv_pic, csv_logs)]
        self.inventory_logs.append(None)
        self.inventory_logs = self.inventory_logs[-self.inventory_logs_max:]
        self.inventory_count_diff = [0 for i in range(len(self.inventory_count))]
        self.inventory_count_temp = self.inventory_count.copy()

    def update_game_params(self):
        # update map
        if(self.map_status and not self.map_status_last):
            self.active_map = self.map_unlocked
            self.active_mask = self.mask_unlocked            
        elif(not self.map_status and self.map_status_last):
            self.active_map = self.map_locked
            self.active_mask = self.mask_locked  
        self.map_status_last = self.map_status          

        # update position
        target_pos = self.char_cur_position.copy()
        posX, posY = target_pos.copy()
        move_L, move_R, move_U, move_D = self.char_move_bool
        # target_pos = self.char_cur_position.copy()
        if(move_L):
            target_pos[0] = posX - 1
        if(move_R):
            target_pos[0] = posX + 1
        if(move_U):
            target_pos[1] = posY - 1
        if(move_D):
            target_pos[1] = posY + 1
        walkable = self.active_mask[target_pos[1], target_pos[0]]
        if(walkable > 0):
            self.char_cur_position = target_pos
    
        # find trigger
        if (posX >= self.trigger_in[0] and posX <= self.trigger_in[2]) and (posY >= self.trigger_in[1] and posY <= self.trigger_in[3]):
            self.trigger_stat_bool = 0
        elif (posX >= self.trigger_out[0] and posX <= self.trigger_out[2]) and (posY >= self.trigger_out[1] and posY <= self.trigger_out[3]):
            self.trigger_stat_bool = 1
        else:
            self.trigger_stat_bool = -1

        # trigger response
        trig_cur, trig_last = self.trigger_stat_bool, self.trigger_stat_bool_last
        if(trig_cur==0 and trig_last==-1):
            pass
            # self.map_status = True
            # self.face_recognition.reset_counter()
            # self.screen_idx = 2
        elif(trig_cur==0 and trig_last==1):
            self.map_status = False
            self.face_output = None
            self.face_recognition.reset_counter()
        elif(trig_cur==-1 and trig_last==0):
            self.map_status = False
            self.face_output = None
            self.face_recognition.reset_counter()

        if(self.face_recognition.detections_final):
            self.map_status = True
            self.personal_belongings = list(self.reset.inventories[self.face_recognition.detections_final].values())
        # update trigger state
        self.trigger_stat_bool_last = self.trigger_stat_bool

        trig_x1, trig_y1, trig_x2, trig_y2 = self.trigger_inventory
        if (posX >= trig_x1 and posX <= trig_x2) and (posY >= trig_y1 and posY <= trig_y2):
            self.trigger_inventory_stat = True
        else:
            self.trigger_inventory_stat = False


    

    def update_inventory_params(self):
        def save_csv():
            filepath = './report.csv'
            csv_time, csv_pic, csv_logs = [], [], []
            with open(filepath) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for i, row in enumerate(csv_reader):
                    if(i > 0):
                        csv_time.append(row[0])
                        csv_pic.append(row[1])
                        csv_logs.append(row[2])
        
            if len(list(set(self.inventory_count_diff))) > 1:
                change = True
            elif(list(set(self.inventory_count_diff))[0] != 0):
                change = True
            else:
                change = False
            
            if(change):
                with open(filepath, "w+") as file:
                    writer = csv.writer(file)
                    writer.writerow(['timestamp', 'PIC', 'logs'])
                    for (time, name, log) in (zip(csv_time, csv_pic, csv_logs)):
                        writer.writerow([time, name, log])
                    writer.writerow([str(datetime.now()).split('.')[0], self.face_recognition.detections_final, self.inventory_count_diff])
                self.load_csv()

        if(self.inventory_save_trigger):
            save_csv()
            self.reset.store_pickle(self.inventory_count, self.personal_belongings, self.face_recognition.detections_final)
            self.inventory_save_trigger = False
            self.screen_idx = 1

        if(self.reset_logs_trigger):
            self.load_csv()
            self.reset_logs_trigger = False

        

    def update_game_screen(self):
        self.screen.fill([0,0,0])
        self.screen.blit(self.active_map, (0,0))
        if(not self.map_status):
            self.screen.blit(self.arrow_image, (903-15, 395-22))
        self.screen.blit(self.char_idle, (self.char_cur_position[0] - 25, self.char_cur_position[1] - 37))
        if(self.map_status):
            text = self.font.render(self.face_recognition.detections_final, True, self.color_red, None)
            textRect = text.get_rect()
            textRect.center = (300, 150)
            self.screen.blit(text, textRect)
        if(self.trigger_inventory_stat):
            text = self.trigger_font.render("--- Press Enter to open the craft ---", True, self.color_red, None)
            textRect = text.get_rect()
            textRect.center = (310, 410)
            self.screen.blit(text, textRect)
        if(self.trigger_stat_bool==0):
            text = self.trigger_font.render("--- Press Enter to identify yourself ---", True, self.color_red, None)
            textRect = text.get_rect()
            textRect.center = (310, 540)
            self.screen.blit(text, textRect)
        

    def update_main_screen(self):
        self.screen.fill([0,0,0])
        self.screen.blit(self.menu_screen, (0,0))        
        strings = ['Run', 'Retrain Model', 'Reset Logs']
        smallfont = pygame.font.SysFont('Corbel',35)  
        self.temp_button_status = np.zeros_like(strings)
        for index, string in enumerate(strings):
            button_text = string
            text = smallfont.render(button_text , True , self.color_white) 
            delta = index*self.distance_between_buttons
            textRect = text.get_rect()
            textRect.center = (self.width/2, self.height+delta)
    
            button_stat = int((self.width/2-self.button_w/2 <= self.mouse[0] <= self.width/2+self.button_w/2) and \
                              (self.height-self.button_h/2+delta <= self.mouse[1] <= self.height+self.button_h/2+delta))
            self.temp_button_status[index] = button_stat
            if button_stat:
                pygame.draw.rect(self.screen, self.color_light, [self.width/2-self.button_w/2, self.height-self.button_h/2+delta, self.button_w, self.button_h])
            else:
                pygame.draw.rect(self.screen, self.color_dark, [self.width/2-self.button_w/2, self.height-self.button_h/2+delta, self.button_w, self.button_h])
            # screen.blit(text, (int(width/2-button_w/2), int(height-button_h/2+delta)))
            self.screen.blit(text, textRect)
        self.button_status = self.temp_button_status

    def update_inference_screen(self):
        ret, frame = self.camera.read()
        self.screen.fill([0,0,0])
        self.screen.blit(self.inference_screen, (0,0))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_orig = frame.copy()
        try:
            frame, lowest_dir, lowest_id, loss = self.face_recognition.detect_faces(frame_orig, draw=True)
        except:
            lowest_dir, lowest_id, loss = None, None, None
        frame = cv2.resize(frame, (667, 500))
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)

        self.screen.blit(frame, (112,72))

        # create a text suface object,
        # on which text is drawn on it.
        percent = f"{int(self.face_recognition.detections_belief * 100)}%"
        text1 = self.font.render(percent, True, self.color_white, None)
        text2 = self.font.render(lowest_dir, True, self.color_white, None)
        textRect1 = text1.get_rect()
        textRect2 = text2.get_rect()
        textRect1.center = (1080, 270)
        textRect2.center = (1080, 320)
        self.screen.blit(text1, textRect1)
        if(self.face_recognition.detections_face is not None):
            self.screen.blit(text2, textRect2)
        # self.face_output = lowest_dir

        if(self.face_recognition.detections_final is not None):
            # self.map_status = True
            self.screen_idx = 1

        text_close = self.trigger_font.render("--- Press q or Enter to close the detection ---", True, self.color_red, None)
        textRect_close = text_close.get_rect()
        textRect_close.center = (640, 620)
        self.screen.blit(text_close, textRect_close)

    def update_inventory_screen(self):
        def find_difference():
            self.inventory_count_diff = [(src1-src2) for (src1, src2) in zip(self.inventory_count, self.inventory_count_temp)]
            if len(list(set(self.inventory_count_diff))) > 1:
                log = self.inventory_count_diff
            elif(list(set(self.inventory_count_diff))[0] != 0):
                log = self.inventory_count_diff
            else:
                log = [0 for i in range(len(self.inventory_count_diff))]
            return log

        def update_counter(idx=0, op_type=0): #1 for add, 0 for substract            
            if(op_type == 1 and (self.inventory_count[idx] + 1 <= self.inventory_count_max) and (self.personal_belongings[idx] - 1 >= 0)):
                self.inventory_count[idx] += 1
                self.personal_belongings[idx] -= 1
                timestamp = str(datetime.now()).split('.')[0]
                logs = find_difference()
                self.inventory_logs[-1] = '{} {} {}'.format(timestamp, self.face_recognition.detections_final, logs)
            elif(op_type == -1 and (self.inventory_count[idx] - 1 >= self.inventory_count_min)):
                self.inventory_count[idx] -= 1
                self.personal_belongings[idx] += 1
                timestamp = str(datetime.now()).split('.')[0]
                logs = find_difference()
                self.inventory_logs[-1] = '{} {} {}'.format(timestamp, self.face_recognition.detections_final, logs)
            
            textRecords = [self.inventory_font_history.render(records, True, self.color_red, None) for records in self.inventory_logs]
            textRectsRecords = [text.get_rect() for text in textRecords]
            for i, (textRec, textRectRec) in enumerate(zip(textRecords, textRectsRecords)):
                button_personal_inventory_w_pos = 600
                width_pos, height_pos = 895, 100 + 30*i
                textRectRec.topleft = (width_pos, height_pos)
                self.screen.blit(textRec, textRectRec)

        add_button_status = (self.inventory_add_button.max()==1)
        sub_button_status = (self.inventory_sub_button.max()==1)
        
        if(self.inventory_change_trigger):
            if(add_button_status):
                update_counter(np.argmax(self.inventory_add_button), 1)
            elif(sub_button_status):
                update_counter(np.argmax(self.inventory_sub_button), -1)
            self.inventory_change_trigger = False

        self.screen.fill([0,0,0])
        self.screen.blit(self.inventory_screen, (0,0))
        button_w_pos = 400
        button_w_pos_delta = 100
        button_w, button_h = 40, 40
        button_personal_inventory_w_pos = 600

        
        inventories = [inventory + ' : ' for inventory in self.inventory_name]
        inventories_count = self.inventory_count
        texts = [self.font.render(inventory + str(inventories_count[i]), True, self.color_red, None) for i, inventory in enumerate(inventories)]
        textRects = [text.get_rect() for text in texts]
        addText = self.button_font_addsub.render('<' , True , self.button_color1_addsub)
        addRect = addText.get_rect()
        subText = self.button_font_addsub.render('>' , True , self.button_color2_addsub)
        subRect = addText.get_rect()
        personal_inventor_texts = [self.font.render(str(stuffs), True, self.color_red, None) for i, stuffs in enumerate(self.personal_belongings)]
        personal_inventor_textrects = [text.get_rect() for text in personal_inventor_texts]

        for (i, (text, textRect, personalText, personalTextRects)) in enumerate(zip(texts, textRects, personal_inventor_texts, personal_inventor_textrects)):
            width_pos, height_pos = 95, 100 + self.distance_between_buttons*i
            textRect.topleft = (width_pos, height_pos)
            personalTextRects.topleft = (button_personal_inventory_w_pos, height_pos)
            add_button_stat = int((button_w_pos <= self.mouse[0] <= button_w_pos + button_w) and \
                                (height_pos <= self.mouse[1] <= height_pos + button_h))
            sub_button_stat = int((button_w_pos + button_w_pos_delta <= self.mouse[0] <= button_w_pos + button_w + button_w_pos_delta) and \
                                (height_pos <= self.mouse[1] <= height_pos + button_h))
            if(add_button_stat):
                self.inventory_add_button[i] = 1
                pygame.draw.rect(self.screen, self.button_color2_addsub, [button_w_pos, height_pos, button_w, button_h])
                addText = self.button_font_addsub.render('<' , True , self.button_color1_addsub)
                addRect = addText.get_rect()
                addRect.center = (button_w_pos + button_w/2, height_pos + button_h/2)
                self.screen.blit(addText, addRect)
            else:
                self.inventory_add_button[i] = 0
                pygame.draw.rect(self.screen, self.button_color1_addsub, [button_w_pos, height_pos, button_w, button_h])
                addText = self.button_font_addsub.render('<' , True , self.button_color2_addsub)
                addRect = addText.get_rect()
                addRect.center = (button_w_pos + button_w/2, height_pos + button_h/2)
                self.screen.blit(addText, addRect)

            if(sub_button_stat):
                self.inventory_sub_button[i] = 1
                pygame.draw.rect(self.screen, self.button_color2_addsub, [button_w_pos + button_w_pos_delta, height_pos, button_w, button_h])
                addText = self.button_font_addsub.render('>' , True , self.button_color1_addsub)
                addRect = addText.get_rect()
                addRect.center = (button_w_pos + button_w_pos_delta + button_w/2, height_pos + button_h/2)
                self.screen.blit(addText, addRect)

            else:
                self.inventory_sub_button[i] = 0
                pygame.draw.rect(self.screen, self.button_color1_addsub, [button_w_pos + button_w_pos_delta, height_pos, button_w, button_h])
                addText = self.button_font_addsub.render('>' , True , self.button_color2_addsub)
                addRect = addText.get_rect()
                addRect.center = (button_w_pos + button_w_pos_delta + button_w/2, height_pos + button_h/2)
                self.screen.blit(addText, addRect)

            self.screen.blit(text, textRect)    
            self.screen.blit(personalText, personalTextRects)  
        update_counter()  

        text_close = self.trigger_font.render("--- Press q or Enter to close the craft ---", True, self.color_red, None)
        textRect_close = text_close.get_rect()
        textRect_close.center = (640, 620)
        self.screen.blit(text_close, textRect_close)

if __name__ == "__main__":
    main()