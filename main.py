import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scanDDM import scanDDM
from vis import draw_scanpath, compute_density_image

sns.set_context("talk")

# prompt -------------------------------------------------------------
prompt = "small sheep on right front"

# image path ---------------------------------------------------------
img_path = "data/sheep.jpg"


# experiment parameters ----------------------------------------------
fps = 25                 #Frames per second
exp_dur = 2.             #Experiment duration (seconds)
n_obs = 100              #number of observers (scanpaths) to simulate
weight_old_objmap = 0.2  #Weight given to the old saliency map
weight_new_objmap = 0.8  #Weight given to the new saliency map


# model parameters ---------------------------------------------------
k = 10                   #Cauchy distribution dispersion
threshold = 1.0          #Race Model threshold
noise = 7                #Race Model diffusion strenght
eta = 17                 #Race Model baseline accumulation
 
 
# initialization
first_iteration = True
x_plot = []
y_plot = []
d_plot = []
all_scans = []
incremental_prompt = ""
old_obj_map = None


# load image ---------------------------------------------------------
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
typical_shape = (768, 1024, 3)

if img.shape != typical_shape:
    if img.shape[0] > img.shape[1]:
        img = cv2.resize(img, (768, 1024))
    else:
        img = cv2.resize(img, (1024, 768))


# prompt preparation  -----------------------------------------------
prompt_splitted = prompt.split()    # splitting the prompt in words
prompt_splitted.insert(0, "")       # BOT
prompt_splitted.append("")          # EOT


# iterate the model for each word -----------------------------------
for i, word in enumerate(prompt_splitted):
        
    # prompt preparation incrementally adding words and spaces between them    
    incremental_prompt += word
    if i < len(prompt_splitted) - 1: 
        incremental_prompt += " "
    
    # model definition ----------------------------------------------
    model = scanDDM(
        experiment_dur=exp_dur,
        fps=fps,
        threshold=threshold,
        noise=noise,
        kappa=k,
        eta=eta,
        device="cpu",
    )    
    
    # simulate ------------------------------------------------------
    # first iteration (BOT), look at the middle of the image
    if first_iteration:
        size = img.shape
        mean = (size[0] // 2, size[1]//2)
        cov = [[0.5, 0], [0, 0.5]]  # diagonal covariance
        y, x = np.random.multivariate_normal(mean, cov, 1).T
        d = 100
        start_point = x.item(), y.item(), d
        first_iteration = False
        
        
    # next iterations: old object map and starting point from the previous iteration
    else:
        scans, prior_map, old_obj_map, start_point = model.simulate_scanpaths(
            image = img,
            prompt = incremental_prompt,
            n_observers = n_obs,
            old_object_map = old_obj_map,
            start_point = start_point,
            weight_new_objmap = weight_new_objmap,
            weight_old_objmap = weight_old_objmap
        )
        
        all_scans.extend(scans)
        
    # data for plot
    x, y, d = start_point
    x_plot.append(x)        # x coordinates of the fixation
    y_plot.append(y)        # y coordinates of the fixation 
    d_plot.append(d)        # duration of the fixation
     
    
    # print of the fixation data
    if(i == 0):
        word = "BOT"
    if(i == len(prompt_splitted) - 1):
        word = "EOT" 
    print(f"{word} -- x:{x} -- y:{y} -- d:{d}")   


all_scans = np.vstack(scans)


# plot ----------------------------------------------------------------
sp_to_plot = 1          #idx of the simulated scanpath to plot

# original image
fig = plt.figure(tight_layout=True, figsize=(15,10))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Original image")

# simulated scanpath over the image
plt.subplot(2, 2, 2)
plt.imshow(img)
draw_scanpath(x_plot, y_plot, d_plot * 1000)
plt.axis("off")
plt.title("Simulated Scanpath")

# saliency over the image
plt.subplot(2, 2, 3)
sal = compute_density_image(all_scans[:, :2], img.shape[:2])
res = np.multiply(img, np.repeat(sal[:,:,None]/np.max(sal),3, axis=2))
res = res/np.max(res)
plt.imshow(res)
plt.axis("off")
plt.title("Generated Saliency ("+str(n_obs)+" scanpaths)")

# saliency map
plt.subplot(2, 2, 4)
plt.imshow(prior_map, cmap='viridis')
plt.axis("off")
plt.title("Saliency map")

fig.suptitle(f"[BOT] {prompt} [EOT]", fontsize=20)

plt.show()
