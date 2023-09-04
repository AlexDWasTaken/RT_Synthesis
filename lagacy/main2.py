import matplotlib
matplotlib.use('Agg')
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import cv2
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_uint, img_as_int
import torch
import torch.nn.functional as F

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator
from animate import normalize_kp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from process import load_checkpoints, headpose_pred_to_degree, get_kp_driving, get_rotation_matrix, keypoint_transformation 
from detection_utils.face.face_detector import FaceDetector
from detection_utils.pupil.pupil_detector import PupilDetector
from projection_utils.Projectors import ScreenProjector, StraightProjector, RefractionProjector

from timeit import default_timer

def init_detector(cap, distance:float):
    _, frame = cap.read()
    fdetector = FaceDetector(frame, original_depth=distance)
    pdetector = PupilDetector(frame)
    count = 0
    
    while not fdetector.initialized or not pdetector.initialized:
        print(f"f: {fdetector.initialized}")
        print(f"p: {pdetector.initialized}")
        count += 1
        _, frame = cap.read()
        fdetector = FaceDetector(frame, original_depth=distance)
        pdetector = PupilDetector(frame)
        if count > 10:
            raise IOError("Failed to init detectors. Please make sure you are in a clean and bright environment.")
        
    return fdetector, pdetector

def get_cutout(cv2_image, square):
    x, y, a = square
    square_image = cv2_image[y:y+a, x:x+a]
    img = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)
    img = imageio.core.asarray(img)
    resized = resize(img, (256, 256))[..., :3]
    return resized

def sharp_rotation(a, b, threshold = 1.5):
    diff = lambda m, n, key: torch.abs(headpose_pred_to_degree(m[key]) - headpose_pred_to_degree(n[key]))
    diff_yaw = diff(a, b, 'yaw')
    diff_pitch = diff(a, b, 'pitch')
    diff_roll = diff(a, b, 'roll')
    return diff_pitch + diff_roll + diff_yaw >= threshold

def imageio_to_cv2(img):
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--remote_video", help="path to remote video")
    parser.add_argument("--viewpoint_video", help="path to the video which simulates the video capture")
    parser.add_argument("--projector", help="Choose which projector to use.")
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to model config")
    parser.add_argument("--cam_config", default='config/cam.yaml')
    parser.add_argument("--checkpoint", default='', help="path to checkpoint to restore")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--no_pitch", default=False, action="store_true", help="Whether to ignore pitch while processing, since the model does not perform well when dealing with pitch.")
    parser.add_argument("--refraction", default=False, action="store_true")
    parser.add_argument("--straight", default=False, action="store_true")

    args = parser.parse_args()
    
    #Open up the camera
    remote_cap = cv2.VideoCapture(args.remote_video)
    local_cap = cv2.VideoCapture(args.viewpoint_video)
    fps = remote_cap.get(cv2.CAP_PROP_FPS) #assuming the fps is same in two videos.
    print("FPS: ", fps)
    if not remote_cap.isOpened():
        raise IOError("failed to open remote video capture")
    else:
        print("Remote cap open successfully.")
        
    if not local_cap.isOpened():
        raise IOError("failed to open local video capture.")
    else:
        print("Local camera opened successfully.")
    
    #load models
    with open(args.cam_config) as cf:
        cam_config = yaml.load(cf)
    with open(args.config) as f:
        config = yaml.load(f)
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')
    cpu = args.cpu
    generator, kp_detector, he_estimator = load_checkpoints(config_path=args.config, checkpoint_path=args.checkpoint, gen='spade', cpu=args.cpu)
    
    #initialize detectors.Assuming same distance.
    remote_face_detector, remote_pupil_detector = init_detector(remote_cap, distance=float(cam_config['distance']))
    local_face_detector, local_pupil_detector = init_detector(local_cap, distance=float(cam_config['distance']))
    
    
    #Reinitialize the camera to go back to the first frame
    remote_cap = cv2.VideoCapture(args.remote_video)
    local_cap = cv2.VideoCapture(args.viewpoint_video)
    
    remote_ret, remote_frame = remote_cap.read()
    local_ret, local_frame = local_cap.read()
    
    remote_pupil_detector.detect_pupil(remote_frame)
    remote_face_detector.detect_face(remote_frame)
    local_pupil_detector.detect_pupil(local_frame)
    local_face_detector.detect_face(local_frame)
    
    #initialize projector, assuming both sides has same picture resolution.
    frame_height = local_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = local_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    intrinsic = np.array(cam_config['intrinsic'])
    intrinsic[-1, -1] = 1
    print("Intrinsic: ", intrinsic)

    if args.refraction:
        print("Using Refraction Projector")
        projector = RefractionProjector(frame_width, frame_height, 0.355, 0.22, intrinsic)
    elif args.straight:
        print("Using Straight projector")
        projector = StraightProjector(frame_width, frame_height, 0.355, 0.22, intrinsic)
    else:
        print("Using default projector")
        projector = ScreenProjector(frame_width, frame_height, 0.355, 0.22, intrinsic)
    no_pitch = args.no_pitch
    if no_pitch: print("Ignoring pitch while generating.")
    
    print("Finished model initialization.")
    
    
    with torch.no_grad():
        prediction = []
        
        remote_cutout = {
            'coords': remote_face_detector.face_frame,
            'img': get_cutout(remote_frame, remote_face_detector.face_frame)
        }
        original_he = he_estimator(torch.tensor(remote_cutout['img'][np.newaxis].astype(np.float32)).permute(0, 3, 1, 2))
        """        
        relative = {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        }"""
        cached_coords = None
        
        driving_frame = np.array([remote_cutout['img']])
        driving = torch.tensor(np.array(driving_frame)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        he_driving_initial = he_estimator(driving[:, :, 0])
        frame = 0
        
        he_source = None
        
        
        while remote_ret and local_ret:
            
            time = default_timer()
            
            print(f"-------------Processing frame {frame}-------------")
            frame += 1
            
            driving_frame = torch.tensor(remote_cutout['img'][np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                driving_frame = driving_frame.cuda()
            he_driving = he_estimator(driving_frame)
            
            if he_source == None or cached_coords != remote_cutout['coords'] or sharp_rotation(he_driving, he_source):
                print("################Changing source.##################")
                cached_coords = remote_cutout['coords']
                source = torch.tensor(remote_cutout['img'][np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                if not cpu:
                    source = source.cuda()
                kp_canonical = kp_detector(source)
                he_source = he_estimator(source)
                kp_source = keypoint_transformation(kp_canonical, he_source, estimate_jacobian)
                kp_driving_initial = keypoint_transformation(kp_canonical, he_driving_initial, estimate_jacobian)
            
            print("local_face_detector.depth: ", local_face_detector.depth)
            
            time1 = default_timer()
            print("Finished headpose estimation. Time elapsed in this step:", time1 - time)
            
            if isinstance(projector, ScreenProjector):
                freeview = projector.calculate(
                    projector.recover_camera_coordinates(local_pupil_detector.viewpoint, local_face_detector.depth),
                    remote_pupil_detector.viewpoint
                )
            elif isinstance(projector, RefractionProjector):
                freeview = projector.calculate(
                    projector.recover_camera_coordinates(local_pupil_detector.viewpoint, local_face_detector.depth),
                    projector.recover_camera_coordinates(remote_pupil_detector.viewpoint, remote_face_detector.depth)
                )
            elif isinstance(projector, StraightProjector):
                freeview = projector.calculate(
                    projector.recover_camera_coordinates(local_pupil_detector.viewpoint, local_face_detector.depth),
                    projector.recover_camera_coordinates(remote_pupil_detector.viewpoint, remote_face_detector.depth),
                    remote_face_detector.depth
                )
            
            yaw = -float(freeview['yaw'])
            pitch = float(freeview['pitch'])
            roll = -float(freeview['roll'])
            print({
                'yaw': yaw,
                'pitch': 0.0 if no_pitch else pitch,
                'roll': roll
            })
            
            time2 = default_timer()
            print("Finished angle calculation. Time elapsed in this step: ", time2 - time1)
            
            kp_driving = keypoint_transformation(kp_canonical, he_driving, estimate_jacobian, free_view=True, yaw=yaw, pitch=pitch, roll=roll)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=True,
                                   use_relative_jacobian=estimate_jacobian, adapt_movement_scale=True)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            #print(out)
            out = imageio_to_cv2(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            #print(out)
            
            time3 = default_timer()
            print("Finished video generation. Time elapsed in this step: ", time3 - time2)
            
            x, y, l = remote_face_detector.latest_face_frame
            resized_overlay = cv2.resize(out, (l, l))
            concatinated = remote_frame.copy()
            #print((concatinated))
            #print(img_as_ubyte(resized_overlay))
            concatinated[y:y+l, x:x+l] = img_as_ubyte(resized_overlay)
            prediction.append(concatinated)
            
            time4 = default_timer()
            print("Finished concatinating frame. Time elapsed in this step: ", time4-time3)
            
            #Read a new frame.
            remote_ret, remote_frame = remote_cap.read()
            local_ret, local_frame = local_cap.read()
            
            if not remote_ret or not local_ret:
                break
            
            tbp = default_timer()
            remote_pupil_detector.detect_pupil(remote_frame)
            local_pupil_detector.detect_pupil(local_frame)
            
            tbf = default_timer()
            local_face_detector.detect_face(local_frame)
            remote_face_detector.detect_face(remote_frame)
            taf = default_timer()
            
            print("predicting pupil spent ", tbf-tbp)
            print("predicting face spent", taf-tbf)
            
            cached_coords = remote_face_detector.face_frame
            remote_cutout = {
                'coords': remote_face_detector.face_frame,
                'img': get_cutout(remote_frame, remote_face_detector.face_frame)
            }
            
            time5 = default_timer()
            print("Finished reading and detecting next frame. Time elapsed in this step: ", time5-time4)
            print(f"Time elapsed in the whole process: {time5 - time}")

        print("Saving...")
        imageio.mimsave('res.mp4', [img_as_ubyte((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in prediction], fps = fps)
        print("saved.")

            
        
        
        

        
        
        

        
    


    
    
    
    
