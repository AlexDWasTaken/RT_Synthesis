import yaml
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
from timeit import default_timer
from time import sleep

import imageio
import cv2
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
torch.backends.cudnn.benchmark = True
torch.jit.enable_onednn_fusion(True)

from process import load_checkpoints, headpose_pred_to_degree, keypoint_transformation, normalize_kp
from detection_utils.face.face_detector import FaceDetector
from detection_utils.pupil.pupil_detector import PupilDetector
from projection_utils.Projectors import ScreenProjector, StraightProjector, RefractionProjector

from concurrent.futures import ThreadPoolExecutor
from torch import nn
import torch.nn.utils.prune as prune
import warnings
warnings.filterwarnings("ignore")

def lian_dan(generator, log_architecture = True):
    print("\033[32m开始炼丹\033[0m")
    if log_architecture:
        with open("model_structure.txt", 'w') as f:
            for module in generator.modules():
                f.write(str(type(module)) + '\n')
            f.write("#################\n")
            f.write(str(generator))
    
    #Spade Decoder
    #deleted G_Middle_0
    
    #prune
    pruned = 0
    prune_conv2d = False
    prune_conv3d = True
    total = 0
    individual = 0
    conv2d = 0
    conv3d = 0
    for module in generator.modules():
        #print(type(module))
        total += 1
        if isinstance(module, ResBlock2d) or isinstance(module, SameBlock2d) or isinstance(module, UpBlock2d) or isinstance(module, DownBlock2d) or isinstance(module, ResBlock3d) or isinstance(module, DownBlock2d):
            module.prune(amount=0.25)
            pruned += 1
            individual += 1
        elif isinstance(module, torch.nn.Conv2d) and prune_conv2d:
            try:
                module.weight_orig = torch.nn.Parameter(module.weight)
                prune.l1_unstructured(module, name='weight', amount=0.01)
                prune.remove(module, 'weight')
                #print("success")
                pruned += 1
                conv2d += 1
            except:
                print("skipping a pruning step.")
        elif isinstance(module, torch.nn.Conv3d) and prune_conv3d:
            try:
                module.weight_orig = torch.nn.Parameter(module.weight)
                prune.l1_unstructured(module, name='weight', amount=0.2)
                prune.remove(module, 'weight')
                #print("success")
                pruned += 1
                conv3d += 1
            except:
                print("skipping a pruning step.")
        elif isinstance(module, torch.nn.modules.container.ModuleList) and True:
            pruned += 1
            for sub in module:
                sub.prune(amount = 0.3, log = True)
        elif isinstance(module, torch.nn.modules.container.Sequential) and True:
            pruned += 1
            for sub in module:
                if isinstance(sub, nn.Conv2d):
                    try:
                        module.weight_orig = torch.nn.Parameter(module.weight)
                        prune.l1_unstructured(module, name='weight', amount=0.1)
                        prune.remove(module, 'weight')
                        #print("success")
                        pruned += 1
                        conv2d += 1
                    except:
                        print("skipping a pruning step.")

        else:
            #print("skipped", module)
            None
    
    print("\033[32mpurned", pruned, "modules\033[0m")
    print("custom modules: ", individual)
    print("conv2d:", conv2d)
    print("conv3d:", conv3d)
    print("total:", total)
    generator.cuda()
    
    """scripted_generator = torch.jit.script(generator)
    #quantization_config = torch.quantization.default_dynamic_qconfig
    dtype = torch.qint8
    quan_model = torch.quantization.quantize_dynamic(scripted_generator, {torch.nn.Linear}, dtype)
    quan_model.cuda()"""
    return generator

def init_detector(cap, distance: float, pad=0.2, buffer=2):
    _, frame = cap.read()
    fdetector = FaceDetector(frame, original_depth=distance, pad_coef=pad, buffer_size=buffer)
    pdetector = PupilDetector(frame, buffer_size=buffer)
    init_count = 0

    while not fdetector.initialized or not pdetector.initialized:
        print(f"face_detector.initialized: {fdetector.initialized}", end=",  ")
        print(f"pupil_detector.initialized: {pdetector.initialized}", end=", ")
        print(f"attempt {init_count}", end='\r', flush=True)
        init_count += 1
        sleep(0.25)
        _, frame = cap.read()
        fdetector = FaceDetector(frame, original_depth=distance, pad_coef=pad, buffer_size=buffer)
        pdetector = PupilDetector(frame, buffer_size=buffer)
        if init_count > 10:
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
                for _ in range(freq - 1):
                    yield False
                yield True

        return gen(freq)


def get_cutout(cv2_image, square):
    """
    Returns a imgio image.
    """
    x, y, a = square
    square_image = cv2_image[y:y + a, x:x + a]
    img = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)
    img = imageio.core.asarray(img)
    resized = resize(img, (256, 256))[..., :3]
    return resized


def sharp_rotation(a, b, threshold=1.5):
    diff = lambda m, n, key: torch.abs(headpose_pred_to_degree(m[key]) - headpose_pred_to_degree(n[key]))
    diff_yaw = diff(a, b, 'yaw')
    diff_pitch = diff(a, b, 'pitch')
    diff_roll = diff(a, b, 'roll')
    return diff_pitch + diff_roll + diff_yaw >= threshold


def imageio_to_cv2(img):
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2_img


def get_freeview(projector, no_pitch, ld, lvp, rd, rvp):
    freeview = None
    if isinstance(projector, ScreenProjector):
        freeview = projector.calculate(
            projector.recover_camera_coordinates(lvp, ld),
            rvp
        )
    elif isinstance(projector, RefractionProjector):
        freeview = projector.calculate(
            projector.recover_camera_coordinates(lvp, ld),
            projector.recover_camera_coordinates(rvp, rd)
        )
    elif isinstance(projector, StraightProjector):
        freeview = projector.calculate(
            projector.recover_camera_coordinates(lvp, ld),
            projector.recover_camera_coordinates(rvp, rd),
            rd
        )
    else:
        raise TypeError("Unsupported projector type.")

    yaw = -float(freeview['yaw'])
    pitch = 0.0 if no_pitch else float(freeview['pitch'])
    roll = -float(freeview['roll'])
    return yaw, pitch, roll


@torch.no_grad()
def prepare(local_frame, remote_value, local_face_detector, local_pupil_detector, estimate_jacobian):
    local_face_detector.detect_face(local_frame)
    local_pupil_detector.detect_pupil(local_frame)

    local_depth = local_face_detector.get_depth()
    local_vp = local_pupil_detector.get_viewpoint()
    remote_depth = remote_value['depth']
    remote_vp = remote_value['viewpoint']

    yaw, pitch, roll = get_freeview(projector, args.no_pitch, local_depth, local_vp, remote_depth, remote_vp)

    print(f"yaw: {yaw:.3f}, pitch: {pitch:.1f}, roll: {roll:.1f}", end='\r', flush=True)

    kp_driving = keypoint_transformation(remote_value['kp_canonical'], remote_value['he_source'], estimate_jacobian,
                                         free_view=True, yaw=yaw, pitch=pitch, roll=roll)

    kp_norm = normalize_kp(kp_source=remote_value['kp_source'], kp_driving=kp_driving,
                           kp_driving_initial=remote_value['kp_source'], use_relative_movement=True,
                           use_relative_jacobian=estimate_jacobian, adapt_movement_scale=True)

    return kp_norm


@torch.no_grad()
def render(generator, kp_norm, remote_value):
    out = generator(remote_value['source'], kp_source=remote_value['kp_source'], kp_driving=kp_norm)  # 0.0032s
    out = imageio_to_cv2(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    x, y, l = remote_value['face_frame']
    resized_overlay = cv2.resize(out, (l, l))

    concatinated = remote_value['frame']
    concatinated[y:y + l, x:x + l] = img_as_ubyte(resized_overlay)

    cv2.imshow("Captured", local_frame)
    cv2.imshow("Realtime Demo", concatinated)
    k = cv2.waitKey(1)


@torch.no_grad()
def generate_cache(total_frame: int, remote_cap, first_frame: np.ndarray, remote_pupil_detector: PupilDetector,
                   remote_face_detector: FaceDetector, kp_detector, he_estimator, cpu=False):
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
        # he_source_backup = he_source.copy()
        # print(he_source['t'])
        kp_source = keypoint_transformation(kp_canonical, he_source, estimate_jacobian)
        # print(he_source['t'])
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--remote_video", help="path to remote video")
    parser.add_argument("--viewpoint_video", help="path to the video which simulates the video capture")
    parser.add_argument("--projector", help="Choose which projector to use.")
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to model config")
    parser.add_argument("--cam_config", default='config/cam.yaml')
    parser.add_argument("--checkpoint", default='', help="path to checkpoint to restore")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--no_pitch", default=False, action="store_true",
                        help="Whether to ignore pitch while processing, since the model does not perform well when dealing with pitch.")
    parser.add_argument("--refraction", default=False, action="store_true")
    parser.add_argument("--straight", default=False, action="store_true")
    parser.add_argument("--pad", default=0.2, type=float)
    parser.add_argument("--result", default='result.mp4')
    parser.add_argument("--always_update_source", default=True, action="store_true")
    parser.add_argument("--update_freq", default=0, type=int)
    parser.add_argument("--generate_cache", default=False, action='store_true')
    parser.add_argument("--save_cache", default='cache.pickle')
    parser.add_argument("--load_cache", default='cache.pickle')
    parser.add_argument("--detector_buffer_size", default=2, type=int)
    parser.add_argument("--cache_generator_checkpoint", default='')

    args = parser.parse_args()
    
    """
    # load models
    with open(args.config) as f:
        config = yaml.load(f)
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')

    generator, kp_detector, he_estimator = load_checkpoints(config_path=args.config, checkpoint_path=args.checkpoint,
                                                            gen='spade', cpu=False)
    print("###############################################################################################")
    count = 0
    pruned = 0
    for module in generator.modules():
        if isinstance(module, ResBlock2d) or isinstance(module, SameBlock2d) or isinstance(module, UpBlock2d) or isinstance(module, DownBlock2d) or isinstance(module, ResBlock3d) or isinstance(module, DownBlock2d):
            module.prune()
            print("pruned.")
            pruned += 1
        else:
            #print("skipped", module)
            count += 1
                        
    print(count)
    print(pruned)
    
    with open("model_structure.txt", 'w') as f:
        f.write(str(generator))
    
    generatorp = generator"""


    force_update = args.always_update_source

    # Open up the camera
    remote_cap = cv2.VideoCapture(args.remote_video)
    fps = remote_cap.get(cv2.CAP_PROP_FPS)
    total_frame = remote_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS: ", fps)
    if not remote_cap.isOpened():
        raise IOError("failed to open remote video capture")
    else:
        print("Remote cap open successfully.")

    # load models
    with open(args.cam_config) as cf:
        cam_config = yaml.load(cf)
    with open(args.config) as f:
        config = yaml.load(f)
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')
    cpu = args.cpu
    
    generator, kp_detector, he_estimator = load_checkpoints(config_path=args.config, checkpoint_path=args.checkpoint,
                                                                gen='spade', cpu=args.cpu, cache_generator=False, original_kp_detector=True)
        
    # initialize detectors.Assuming same distance.
    remote_face_detector, remote_pupil_detector = init_detector(remote_cap, distance=float(cam_config['distance']),
                                                                pad=args.pad, buffer=args.detector_buffer_size)

    # Reinitialize the camera to go back to the first frame
    remote_cap = cv2.VideoCapture(args.remote_video)

    remote_ret, remote_frame = remote_cap.read()

    remote_pupil_detector.detect_pupil(remote_frame)
    remote_face_detector.detect_face(remote_frame)

    # initialize projector, assuming both sides has same picture resolution.
    frame_height = remote_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = remote_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    intrinsic = np.array(cam_config['intrinsic'])
    print("Intrinsic: ", intrinsic)

    # manage update frequency
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

    # Manage cache
    if args.generate_cache:
        if args.cache_generator_checkpoint != '':
            cache_generator_checkpoint = args.cache_generator_checkpoint
            print(f"Using {cache_generator_checkpoint} to generate cache.")
            _, kpd, hee = load_checkpoints(config_path=args.config, checkpoint_path=cache_generator_checkpoint, gen='spade', cpu=args.cpu, cache_generator=True)
            cache = generate_cache(total_frame, remote_cap, remote_frame, remote_pupil_detector, remote_face_detector, kpd, hee)
        else:
            print("use the same kp_detector and he_estimator")
            cache = generate_cache(total_frame, remote_cap, remote_frame, remote_pupil_detector, remote_face_detector, kp_detector, he_estimator)
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

    local_face_detector, local_pupil_detector = init_detector(local_cap, distance=float(cam_config['distance']),
                                                              pad=args.pad, buffer=args.detector_buffer_size)

    # Do JIT stuff
    ret, local_frame = local_cap.read()
    if not ret: raise IOError("failed to read from camera.")
    render(generator, cache[0]['kp_source'], cache[0])

    print("Projectors initialized.")
    benchmark_result = []
    count = 0
    pool = ThreadPoolExecutor(max_workers=2)
    render_future = None
    start_time = default_timer()

    while True:
        count += 1

        # Read from cam
        ret, local_frame = local_cap.read()
        if not ret: raise IOError("failed to read from camera.")

        frame_count = int((default_timer() - start_time) * fps)
        k = cv2.waitKey(1)
        if k != -1:
            print('exiting...')
            pool.shutdown()
        if frame_count >= len(cache):
            print("\n" + "Reached the end of the video.")
            pool.shutdown()
            break

        remote_value = cache[frame_count]

        kp_norm_task = pool.submit(lambda args: prepare(*args), (
        local_frame, remote_value, local_face_detector, local_pupil_detector, estimate_jacobian))

        render_task = pool.submit(lambda args: render(*args), (generator, kp_norm_task.result(), remote_value))

    end_time = default_timer()
    cv2.destroyAllWindows()
    print("Frames rendered:", count)
    print("FPS: ", count / (end_time - start_time))
    print("SPF: ", (end_time - start_time) / count)
            
    
