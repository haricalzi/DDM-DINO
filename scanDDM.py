import numpy as np
from pixel_race_mcDDM import race_DDM
from zs_obj_ground import get_obj_grounding_salmap
import torchvision.transforms as T
import torch
import warnings

warnings.filterwarnings("ignore")

class scanDDM(object):

    def __init__(self, experiment_dur, fps, task_driven=True, ffi=True, threshold=1., ndt=5, noise=0.1, kappa=10, eta=7, device=None):
        self.DDM_sigma = noise
        self.k = kappa  # kappa, to weight saccades lenght
        self.fps = fps  # Video frames per second
        self.threshold = threshold
        self.ndt = ndt  # Non Decision Time (ms)
        self.eta = eta
        self.ffi = ffi
        self.exp_dur = experiment_dur
        self.task_driven = task_driven
        if device is None:
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def simulate_scanpaths(
        self,
        n_observers,
        image=None,
        prompt=None,
        old_object_map=None,
        start_point=None,
        weight_new_objmap=None,
        weight_old_objmap=None
        ):
        
        scans = []
            
        assert image is not None; "Please provide an image or a precomputed saliency map"
        # Task driven attention
        assert prompt is not None; "Please provide an item to search as a prompt"
        # Computing Task oriented Saliency
        obj_map = get_obj_grounding_salmap(prompt, image)
        
        # sum with the old saliency map
        if old_object_map is not None and obj_map.shape == old_object_map.shape:
            obj_map = (old_object_map * weight_old_objmap) + (obj_map * weight_new_objmap)
            obj_map /= np.max(obj_map)
        
        # save the old object map for the next iteration
        old_object_map = obj_map

        obj_map = torch.tensor(obj_map[None, None, :, :], device=self.device)
        saliency_map = obj_map

        reshaped_saliency = T.Resize(size=15)(saliency_map).squeeze()
        reshaped_saliency = reshaped_saliency / torch.max(reshaped_saliency) + torch.finfo(torch.float32).eps

        img_size = saliency_map.squeeze().shape
        nFrames = int(self.fps * self.exp_dur)
        dwnsampled_size = reshaped_saliency.shape
        ratio = np.array(img_size[:2]) / dwnsampled_size
        
        # Simulating Scanpaths
        for s in range(n_observers):

            # Start point given by the last fixation coordinates
            x, y, d = start_point
            # Conversion from 1028*764 to 15*15 due to resize
            original_width, original_height = 764, 1028
            resized_width, resized_height = 15, 15
            x_scaled = x * resized_width / original_width
            y_scaled = y * resized_height / original_height
            x = np.clip(x_scaled, 0, resized_width - 1)
            y = np.clip(y_scaled, 0, resized_height - 1)

            curr_fix = (int(x), int(y))
            prev_fix = (None, None)
            scan = [curr_fix]
            durs = []
            for iframe in range(nFrames):
                if curr_fix != (None, None):
                    if prev_fix != curr_fix:
                        race_model = race_DDM(winner=curr_fix, fps=self.fps, downsampled_size=dwnsampled_size, threshold=self.threshold, noise=self.DDM_sigma, kappa=self.k, eta=self.eta, ffi=self.ffi, device=self.device,)
                        
                curr_fix, prev_fix_dur, rls = race_model.simulate_race(reshaped_saliency)

                if curr_fix != (None, None):
                    scan.append(np.array(curr_fix))
                    durs.append(prev_fix_dur)

            scan_np = np.vstack(scan)[:-1]
            scan_np = np.flip(scan_np, 1)
            durs_np = np.array(durs)
            scan_np[:, 0] = scan_np[:, 0] * ratio[0] + np.random.normal(20, 10, len(scan_np))
            scan_np[:, 1] = scan_np[:, 1] * ratio[1] + np.random.normal(20, 10, len(scan_np))
            scan_dur = np.hstack([scan_np, durs_np[:, None]])
            scans.append(scan_dur)

        # saving the next starting point
        sp_to_plot = 1
        fix_x = scans[sp_to_plot][:, 0]
        fix_y = scans[sp_to_plot][:, 1]
        fix_d = scans[sp_to_plot][:, 2] * 1000
        if len(fix_x) > 0:                  # if there are fixations, take the last one
            x = fix_x[len(fix_x) - 1]
            y = fix_y[len(fix_x) - 1]
            d = fix_d[len(fix_x) - 1]
        else:
            x, y, d = start_point           # otherwise x, y, d from the previous start point
        start_point = x, y, d


        return scans, saliency_map.cpu().detach().numpy().squeeze(), old_object_map, start_point