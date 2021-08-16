import os
import cv2
import glob
import math
import pickle
import mediapipe as mp

class FaceRecognition():
    def __init__(self, savedmodel=os.path.join('.', 'identities.pickle'), datadir=os.path.join('.', 'data')):
        # check saved model
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.detections_list = []
        self.detections_length = 100
        self.detections_face = None
        self.detections_belief = 0.0
        self.detections_belief_thresh = 0.5
        self.detections_final_thresh = 0.95
        self.detections_final = None
        self.detection_face_loss_thresh = 10

        if os.path.isfile(savedmodel):
            print ("Savedmodel exist")
            self.model = self.load_landmark_dict(savedmodel)
        elif os.path.isdir(datadir):
            print ("Data directory exist")
            dir_list = os.listdir(datadir)
            if(len(dir_list) > 0): #create case for 1 folder only
                print ("Train data exist")
                self.model = self.create_landmark_dict(datadir)
            else:
                assert ("No model exist and no training data provided")
                input("Press any key to continue...")

    def load_landmark_dict(self, pickle_path):
        with open(pickle_path, 'rb') as handle:
            model = pickle.load(handle)
        return model

    def generate_landmark(self, image, draw=False):
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(image)
            x_lm = [lm.x for lm in results.multi_face_landmarks[0].landmark]
            y_lm = [lm.y for lm in results.multi_face_landmarks[0].landmark]
            min_x, max_x, min_y, max_y = min(x_lm), max(x_lm), min(y_lm), max(y_lm)
            x_lm = [(lm-min_x)/(max_x-min_x) for lm in x_lm]
            y_lm = [(lm-min_y)/(max_y-min_y) for lm in y_lm]
        if(draw):
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
            return annotated_image, x_lm, y_lm
        else:
            return x_lm, y_lm

    def create_landmark_dict(self, data_root=os.path.join('.', 'data'), saved_dict=None):
        dir_list = os.listdir(data_root)
        file_list = []
        types = ('*.jpg', '*.jpeg', '*.png') # the tuple of file types
        for dataroot in dir_list:
            for file_type in types:
                file_list.extend(glob.glob(os.path.join(data_root, dataroot, file_type)))

        for idx, file in enumerate(file_list):
            img_type, img_take = os.path.basename(os.path.dirname(file)), os.path.basename(file).split('.')[0]
            image = cv2.imread(file)

            # Convert the BGR image to RGB before processing.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x_lm, y_lm = self.generate_landmark(image)
            new_dict = {img_type : {img_take : {'x' : x_lm, 'y' : y_lm}}}

            if(saved_dict != None):
                if(img_type in saved_dict.keys()):
                    if(img_take in saved_dict[img_type].keys()):
                        saved_dict[img_type][img_take].update(new_dict[img_type][img_take])
                    else:
                        saved_dict[img_type].update(new_dict[img_type])
                else:
                    saved_dict.update(new_dict)
            else:
                saved_dict = new_dict
        savedmodel=os.path.join('.', 'identities.pkl')
        with open(savedmodel, 'wb') as handle:
            pickle.dump(saved_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return saved_dict

    def detect_faces(self, image=None, landmarks=[None, None], draw=False):
        lowest_dir, lowest_id, loss = None, None, None
        try:
            img, x_lm, y_lm = self.generate_landmark(image, draw)
        except:
            x_lm, y_lm = landmarks
        for face_dir in self.model:
            for face_id in self.model[face_dir]:
                x, y = self.model[face_dir][face_id]['x'], self.model[face_dir][face_id]['y']
                error = sum([math.hypot(x_i - xlm_i, y_i - ylm_i) for x_i, y_i, xlm_i, ylm_i in zip(x, y, x_lm, y_lm)])
                if((lowest_dir==None and lowest_id==None and loss==None) or (loss > error)):
                    lowest_dir, lowest_id, loss = face_dir, face_id, error
        self.update_detections(lowest_dir, loss)
        lowest_dir = self.detections_face
        if(draw):
            return img, lowest_dir, lowest_id, loss
        return lowest_dir, lowest_id, loss

    def update_detections(self, name, loss):
        if(len(self.detections_list) >= self.detections_length):
            self.detections_list.pop(0)
        if(loss > self.detection_face_loss_thresh):
            self.detections_list.append('Unknown')
        else:
            self.detections_list.append(name)
        mode = max(set(self.detections_list), key=self.detections_list.count)
        count = self.detections_list.count(mode)
        belief = count / self.detections_length
        
        if(belief > self.detections_belief_thresh and belief < self.detections_final_thresh):
            self.detections_face = mode
            self.detections_final = None
        elif(belief > self.detections_belief_thresh and belief > self.detections_final_thresh):
            if(mode != 'Unknown'):
                self.detections_final = mode
        else:
            self.detections_face = "Unknown"
            self.detections_final = None
        self.detections_belief = belief
        
    def reset_counter(self):
        self.detections_list = []
        self.detections_final = None
