import torch
import cv2
import numpy as np
import matplotlib.patches as patches
from scipy.ndimage import zoom, gaussian_filter
from transformers import AutoModelForZeroShotObjectDetection, GroundingDinoProcessor


def get_obj_grounding_salmap(prompt, img):
    
    # definition of the model
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the model and the processor
    processor = GroundingDinoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # prompt preparation (remember the dot at the end of the sentence)
    prompt += "." 
    text_labels = [prompt] 
    
    # computation with the model
    inputs = processor(images=img, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # post-processing
    target_sizes = torch.tensor([img.shape[:2]])
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold = 0.3,          
        text_threshold = 0.2,         
        target_sizes=target_sizes
    )

    result = results[0]
    
    # new empty salmap
    salmap = np.zeros([768, 1024], dtype=np.float32)
    
    # iteration for each object detected
    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=3, edgecolor='r', facecolor='none')
        #print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
        
        # ellipse in the object box
        mask_ellipse = np.zeros_like(salmap)
        center_x = int((x_min + x_max) // 2)
        center_y = int((y_min + y_max) // 2)
        axes_x = int((x_max - x_min) // 2)
        axes_y = int((y_max - y_min) // 2)
        mask_ellipse = cv2.ellipse(mask_ellipse, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 1, -1)
        salmap += mask_ellipse
        salmap[salmap>=1] = 1   # avoid overlaps
    
    
    # zoom to the original img size
    salmap = zoom(
        salmap,
        [img.shape[0] / salmap.shape[0], img.shape[1] / salmap.shape[1]],
        order=1,
    )
    
    # smoothing
    sigma = 2*(1 / 0.039)
    salmap = gaussian_filter(salmap, sigma=sigma)
    
    # normalization
    salmap /= np.max(salmap)
    
    return salmap
