import os
import cv2
import yaml
import numpy as np
import onnxruntime as ort
from skimage import transform as trans
# from insightface.face_analysis import FaceAnalysis # REMOVED
from utils import preprocess, postprocess           # ### RESTORED ###
import warnings
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress warnings
ort.set_default_logger_severity(3)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration constants
RETINAFACE_CONFIG = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False
}

# Fixed thresholds and parameters
RETINAFACE_CONF_THRESH = 0.5
RETINAFACE_NMS_THRESH = 0.4 
ALIGNMENT_SIZE = 224

ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# ### ADDED PREPROCESS FUNCTION (for new Buffalo model) ###
# ### RENAMED to avoid conflict ###
def preprocess_buffalo(img, input_size=(640, 640)):
    """
    Preprocesses the image for the new Buffalo ONNX model.
    - Resizes with aspect ratio preservation
    - Pads to input_size
    - Normalizes
    - Transposes to (B, C, H, W)
    - Swaps BGR to RGB
    
    Returns:
        blob (np.ndarray): The processed image tensor
        det_scale (float): The scale factor to convert results back
    """
    
    # Get original image shape
    h_orig, w_orig = img.shape[:2]
    
    # Calculate new size maintaining aspect ratio
    im_ratio = float(h_orig) / w_orig
    model_ratio = float(input_size[1]) / input_size[0] # H/W
    
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    
    det_scale = float(new_height) / h_orig
    
    resized_img = cv2.resize(img, (new_width, new_height))
    
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img
    
    # --- Preprocessing ---
    det_img_rgb = det_img[..., ::-1] # BGR to RGB
    blob = (det_img_rgb.astype(np.float32) - 127.5) / 128.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    
    return blob, det_scale

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("Configuration loaded successfully")
    return config

def init_session(onnx_model_path, device):
    """Initialize ONNX Runtime session"""
    if device.lower() == "cuda":
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    return session

def estimate_norm(lmk, image_size=224):
    """Estimate transformation matrix for face alignment"""
    ratio = image_size / 112.0 if image_size % 112 == 0 else image_size / 128.0
    diff_x = 0 if image_size % 112 == 0 else 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]

def align_and_crop(img, landmark, image_size=224):
    """Align and crop face using landmarks"""
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

def load_detector(detector_type, device):
    """Load face detector based on type"""
    
    if detector_type == "buffalo":
        # NOTE: This loads your ONNX model with built-in post-processing.
        model_path = "models/scfrd_640_with_postprocessing2.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Buffalo model not found at: {model_path}. "
                                    "Please ensure your new model (with post-processing) is at this location.")
        sess = init_session(model_path, device)
        print("Buffalo (with post-processing) detector loaded successfully")
        return {"type": "buffalo", "detector": sess}
    
    elif detector_type == "retinaface":
        sess = init_session("models/retinaface_detector.onnx", device)
        print("RetinaFace detector loaded successfully")
        return {"type": "retinaface", "detector": sess}
    
    raise ValueError(f"Unknown detector type: {detector_type}")

def detect_faces(detector_model, img, device):
    """Detect faces in image using specified detector"""
    print("Performing detection...")
    
    if detector_model["type"] == "buffalo":
        sess = detector_model["detector"]
        
        # 1. Preprocess the image (using our new function)
        blob, det_scale = preprocess_buffalo(img, input_size=(640, 640))
        
        # 2. Define dynamic thresholds
        score_thresh_val = np.array([RETINAFACE_CONF_THRESH], dtype=np.float32)
        iou_thresh_val = np.array([RETINAFACE_NMS_THRESH], dtype=np.float32)

        # 3. Get input names and run inference
        input_names = [i.name for i in sess.get_inputs()]
        main_input_name = input_names[0]
        
        outputs = sess.run(None, { 
            main_input_name: blob, 
            "score_threshold_input": score_thresh_val,
            "iou_threshold_input": iou_thresh_val
        })
        
        # 4. Unpack and scale results
        boxes, scores, kps = outputs[0], outputs[1], outputs[2]

        if boxes.shape[0] == 0:
            return []
            
        # 5. Scale results back to original image size
        boxes /= det_scale
        kps /= det_scale
        
        # Reshape keypoints
        kps = kps.reshape((kps.shape[0], -1, 2))
        
        # 6. Format output to match other branch
        detections = []
        for i in range(boxes.shape[0]):
            det = {
                "bbox": boxes[i],
                "kps": kps[i],
                "det_score": scores[i]
            }
            detections.append(det)
        
        return detections
    
    else:  # retinaface
        # ### RESTORED ORIGINAL RETINAFACE LOGIC ###
        sess = detector_model["detector"]
        # 'preprocess' and 'postprocess' are now imported from 'utils.py'
        img_in, scale, resize = preprocess(img, [640, 640], device)
        outputs = sess.run(None, {sess.get_inputs()[0].name: img_in})
        dets = postprocess(
            RETINAFACE_CONFIG, img, outputs, scale, resize,
            RETINAFACE_CONF_THRESH, RETINAFACE_NMS_THRESH, 
            device, [640, 640]
        )
        if dets.shape[0] == 0:
            return []
        # Format output to match the buffalo branch
        return [{"bbox": det[:4], "kps": det[5:].reshape(5, 2), "det_score": det[4]} for det in dets]

def _process_single_image(img_path, detector_model, device):
    """Process single image: detect faces, align and crop"""
    print(f"\nProcessing image: {img_path}")
    
    if not os.path.exists(img_path):
        print(f"ERROR: Image file not found: {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not read image: {img_path}")
        return
    print(f"Image loaded successfully - Shape: {img.shape}")

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = f"output_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
    cv2.imwrite(original_path, img)
    print(f"Original image saved: {original_path}")

    print("\nStarting face detection...")
    detections = detect_faces(detector_model, img, device)
    
    if not detections:
        print("No faces detected in the image")
        return
    
    print(f"Successfully detected {len(detections)} face(s)")
    
    detection_info = []
    
    # ### This loop now works for both detectors ###
    for idx, det in enumerate(detections):
        
        bbox = det["bbox"]
        landmarks = det["kps"]
        det_score = det["det_score"]
              
        x1, y1, x2, y2 = map(int, bbox)
        
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(img.shape[0], y2)
        x2 = min(img.shape[1], x2)
        
        cropped_face = img[y1:y2, x1:x2]
        
        #cropped_path = os.path.join(output_dir, f"face_{idx}_cropped.jpg")
        #cv2.imwrite(cropped_path, cropped_face)
        #print(f"Detection saved: {cropped_path}")
        
        print(f"\nProcessing detected face {idx + 1}/{len(detections)}")
        print("Performing face alignment and cropping...")
        aligned_img = align_and_crop(img, landmarks, ALIGNMENT_SIZE)
        
        aligned_path = os.path.join(output_dir, f"face_{idx}_aligned_{ALIGNMENT_SIZE}x{ALIGNMENT_SIZE}.jpg")
        cv2.imwrite(aligned_path, aligned_img)
        print(f"Aligned face saved: {aligned_path}")
        
        face_info = {
            "face_idx": idx,
            "bbox": bbox.tolist(),
            "landmarks": landmarks.tolist(),
            "det_score": float(det_score),
            #"detection_output": cropped_path,
            "aligned_path": aligned_path
        }
        detection_info.append(face_info)
    
    print("\nSaving metadata...")
    metadata_path = os.path.join(output_dir, "detection_results.json")
    metadata = {
        "image_path": img_path,
        "image_shape": img.shape,
        "detector_type": detector_model["type"],
        "device": device,
        "alignment_size": ALIGNMENT_SIZE,
        "num_faces_detected": len(detections),
        "faces": detection_info
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
def process_single_image():
    """Main processing pipeline"""
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        print("Please create a config.yaml file with image_path, detector, and device")
        return
    
    config = load_config(config_path)
    
    required_keys = ['image_path', 'detector', 'device']
    for key in required_keys:
        if key not in config:
            print(f"ERROR: Missing required config parameter: {key}")
            return

    detector_model = load_detector(config['detector'], config['device'])
    
    _process_single_image(config['image_path'], detector_model, config['device'])