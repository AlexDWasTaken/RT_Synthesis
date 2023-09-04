import matplotlib
matplotlib.use('Agg')
import yaml
import pickle
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import cv2
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float
import torch

from process import load_checkpoints, headpose_pred_to_degree, keypoint_transformation, normalize_kp
from process import FaceDetectorThread, PupilDetectorThread, FaceDetectorProcess, PupilDetectorProcess
from detection_utils.face.face_detector import FaceDetector
from detection_utils.pupil.pupil_detector import PupilDetector
from projection_utils.Projectors import ScreenProjector, StraightProjector, RefractionProjector

from timeit import default_timer

def init_detector(cap, distance:float, pad = 0.2):
    _, frame = cap.read()
    fdetector = FaceDetector(frame, original_depth=distance, pad_coef=pad)
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

def get_update_manager(freq):
    if freq == 0:
        
        def gen():
            while True:
                yield False
        
        return gen()
    else:
        
        def gen(freq):
            while True:
                for _ in range(freq-1):
                    yield False
                yield True
        
        return gen(freq)

def get_cutout(cv2_image, square):
    """
    Returns a imgio image.
    """
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

@torch.no_grad()
def generate_cache(total_frame: int, remote_cap, first_frame: np.ndarray, remote_pupil_detector: PupilDetector, remote_face_detector: FaceDetector, cpu=False):
    cache = []
    remote_frame = first_frame
    for _ in tqdm(range(int(total_frame)), desc='Progress'):
        """recognition & depth estimation"""
        remote_pupil_detector.detect_pupil(remote_frame)
        remote_face_detector.detect_face(remote_frame)
        viewpoint = remote_pupil_detector.viewpoint
        depth = remote_face_detector.depth
        
        """model"""
        remote_cutout = {
            'coords': remote_face_detector.face_frame,
            'img': get_cutout(remote_frame, remote_face_detector.face_frame)
        }
        source = torch.tensor(remote_cutout['img'][np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        kp_canonical = kp_detector(source)
        he_source = he_estimator(source)
        #he_source_backup = he_source.copy()
        #print(he_source['t'])
        kp_source = keypoint_transformation(kp_canonical, he_source, estimate_jacobian)
        #print(he_source['t'])
        # kp_driving = kp_canonical
        # he_driving = he_source
        # kp_driving_initial = kp_driving
        # kp_driving
        
        
        cached_value = {
            'frame': remote_frame,
            'source': source,
            'kp_canonical': kp_canonical,
            'he_source': he_source,
            'kp_source': kp_source,
            'viewpoint': viewpoint,
            'depth': depth,
            'face_frame': remote_face_detector.latest_face_frame
        }
        
        cache.append(cached_value)
    
        remote_ret, remote_frame = remote_cap.read()
        if not remote_ret: break
        
    return cache
    

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
    parser.add_argument("--pad", default=0.2, type=float)
    parser.add_argument("--result", default='result.mp4')
    parser.add_argument("--always_update_source", default=True, action="store_true")
    parser.add_argument("--update_freq", default=0, type=int)
    parser.add_argument("--generate_cache", default=False, action='store_true')
    parser.add_argument("--save_cache", default='cache.pickle')
    parser.add_argument("--load_cache", default='cache.pickle')

    args = parser.parse_args()
    
    force_update = args.always_update_source
    
    #Open up the camera
    remote_cap = cv2.VideoCapture(args.remote_video)
    fps = remote_cap.get(cv2.CAP_PROP_FPS) 
    total_frame = remote_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS: ", fps)
    if not remote_cap.isOpened():
        raise IOError("failed to open remote video capture")
    else:
        print("Remote cap open successfully.")
    
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
    remote_face_detector, remote_pupil_detector = init_detector(remote_cap, distance=float(cam_config['distance']), pad = args.pad)
    
    
    #Reinitialize the camera to go back to the first frame
    remote_cap = cv2.VideoCapture(args.remote_video)
    
    remote_ret, remote_frame = remote_cap.read()
    
    remote_pupil_detector.detect_pupil(remote_frame)
    remote_face_detector.detect_face(remote_frame)
    
    #initialize projector, assuming both sides has same picture resolution.
    frame_height = remote_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = remote_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    intrinsic = np.array(cam_config['intrinsic'])
    print("Intrinsic: ", intrinsic)

    #manage update frequency
    update_manager = get_update_manager(args.update_freq)        
    
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
    
    #Manage cache
    if args.generate_cache:
        cache = generate_cache(total_frame, remote_cap, remote_frame, remote_pupil_detector, remote_face_detector)
        if args.save_cache:
            with open(args.save_cache, 'wb') as file:
                print("Saving cache...")
                pickle.dump(cache, file)
                print("Cache saved at ", args.save_cache)
    else:
        with open(args.load_cache, 'rb') as file:
            print("Reading cache from ", args.load_cache)
            cache = pickle.load(file)
            if cache[1] is not None:
                print("finished.")
            else:
                print("An error occurred while reading from", args.load_cache)
    
    # manage realtime generation
    
    local_cap = cv2.VideoCapture(0)
    local_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    local_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not local_cap.isOpened():
        raise IOError("failed to open local video capture.")
    else:
        print("Local camera opened successfully.")
    
    local_face_detector, local_pupil_detector = init_detector(local_cap, distance=float(cam_config['distance']), pad = args.pad)
    
    count = 0
    
    start_time = default_timer()
    
    while True:
        
        ret, local_frame = local_cap.read()
        
        if not ret: raise IOError("failed to read from camera.")
        
        count += 1
        
        frame_count = int((default_timer() - start_time) * fps)
        
        if frame_count >= len(cache): 
            print("Reached the end of the video.")
            break
        
        remote_value = cache[frame_count]
        
        """local_face_detection_process = FaceDetectorProcess(local_face_detector, local_frame)
        local_pupil_detector_process = PupilDetectorProcess(local_pupil_detector, local_frame)
        
        local_face_detection_process.start()
        local_pupil_detector_process.start()
        local_pupil_detector_process.join()
        local_face_detection_process.join()"""
        
        
        
        local_face_detector.detect_face(local_frame)
        local_pupil_detector.detect_pupil(local_frame)
        
        
        if isinstance(projector, ScreenProjector):
                freeview = projector.calculate(
                    projector.recover_camera_coordinates(local_pupil_detector.viewpoint, local_face_detector.depth),
                    remote_value['viewpoint']
                )
        elif isinstance(projector, RefractionProjector):
            freeview = projector.calculate(
                projector.recover_camera_coordinates(local_pupil_detector.viewpoint, local_face_detector.depth),
                projector.recover_camera_coordinates(remote_value['viewpoint'], remote_value['depth'])
            )
        elif isinstance(projector, StraightProjector):
            freeview = projector.calculate(
                projector.recover_camera_coordinates(local_pupil_detector.viewpoint, local_face_detector.depth),
                projector.recover_camera_coordinates(remote_value['viewpoint'], remote_value['depth']),
                remote_value['depth']
            )
        else:
            raise TypeError("Unsupported projector type.")
        
        yaw = -float(freeview['yaw'])
        pitch = 0.0 if args.no_pitch else float(freeview['pitch'])
        roll = -float(freeview['roll'])
        print({
            'yaw': yaw,
            'pitch': 0.0 if args.no_pitch else pitch,
            'roll': roll
        })
        
        kp_driving = keypoint_transformation(remote_value['kp_canonical'], remote_value['he_source'], estimate_jacobian, 
                                             free_view=True, yaw=yaw, pitch=pitch, roll=roll)
        
        kp_norm = normalize_kp(kp_source=remote_value['kp_source'], kp_driving=kp_driving,
                               kp_driving_initial=remote_value['kp_source'], use_relative_movement=True,
                               use_relative_jacobian=estimate_jacobian, adapt_movement_scale=True)
        
        out = generator(remote_value['source'], kp_source = remote_value['kp_source'], kp_driving = kp_norm)
        out = imageio_to_cv2(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        
        x, y, l = remote_value['face_frame']
        resized_overlay = cv2.resize(out, (l, l))
        print(l)
        
        concatinated = remote_value['frame']
        concatinated[y:y+l, x:x+l] = img_as_ubyte(resized_overlay)

        cv2.imshow("Captured", local_frame)
        cv2.imshow("Realtime Demo", concatinated)
        k = cv2.waitKey(1)
        if k != -1: break
    
    end_time = default_timer()
    cv2.destroyAllWindows()
    print("Frames rendered:", count)
    print("FPS: ", count / (end_time - start_time))
    print("SPF: ", (end_time - start_time) / count)

        
        
            
        
    
            
            
            
        

                