from datasetReader import datasetReader
import pandas as pd
import numpy as np
import metrics_definition as sm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image
from tqdm import tqdm


# parameters for the metrics computation
dR = datasetReader("results/refcocogaze_val_correct.json")                      # path human dataset       
ref_width = 1680                                                                # width of the image used for human scanpaths
ref_height = 1050                                                               # height of the image used for human scanpaths                                           
df_simulated = pd.read_csv("results/experiments/results_scanDDM.csv")           # path of csv for simulated scanpaths
width_x = 1024                                                                  # width of the image used for simulated scanpaths                                    
height_y = 768                                                                  # height of the image used for simulated scanpaths
scale = 400                                                                     # scale factor used in some metrics                                       
max_step = 100                                                                  # max length for comparison in SS metric    


"""
df_simulated = pd.read_csv(results/experiments/results_ART.csv")
width_x = 512
height_y = 320

df_simulated = pd.read_csv("results/experiments/results_DDM-DINO.csv")
width_x = 1024
height_y = 768
"""


# normalized dataframe
columns_names = ['ID','norm_X','norm_Y']
normalized_df=pd.DataFrame(columns=columns_names)
normalized_df['ID']= df_simulated['ID']
normalized_df['norm_X']= df_simulated['X'] / width_x
normalized_df['norm_Y']= df_simulated['Y'] / height_y

# list of ids
img_ids = dR.get_dataset_image_id()

# lists used for compute the metrics
euclidean_list = []
string_edit_distances_list = []
scanMatch_list = []

# results dataframe with all the metrics
columns_names = ['ID', 'Euclidean', 'StringEditDistance','ScanMatch', 'SequenceScore']
metrics_results_df = pd.DataFrame(columns=columns_names)

# creation of the clusters structure from human data to compute SS
clusters = {}
for img_id_human in tqdm(img_ids, desc="Clustering Human Scanpaths"):
    human_scanpaths_list = dR.get_human_scanpath(img_id_human, normalized=True, image_width=ref_width, image_height=ref_height)
    if human_scanpaths_list:
        all_coords = np.concatenate([h_scanpath for h_scanpath in human_scanpaths_list], axis=0)
        if len(all_coords) > 0:
            bandwidth = estimate_bandwidth(all_coords, n_jobs=-1)
            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(all_coords)
            cluster_strings = {}
            for i, h_scanpath in enumerate(human_scanpaths_list):
                scaled_h_scanpath = h_scanpath
                if len(scaled_h_scanpath) > 0:
                    predicted_clusters = ms.predict(scaled_h_scanpath).tolist()
                    cluster_strings[f'human_{img_id_human}_{i}'] = predicted_clusters
            prompt = dR.get_prompt_from_id(img_id_human).replace(" ", "_")
            clusters[f'test-simulated-{prompt}-{img_id_human}'] = {'cluster': ms, 'strings': cluster_strings}


# resample two scanpaths to a target length using DTW alignment
def resample_scanpath_dtw(sp1, sp2, target_length):
    """
    Args:
        sp1: First scanpath as numpy array of shape (n, 2)
        sp2: Second scanpath as numpy array of shape (m, 2)
        target_length: Desired length for both resampled scanpaths
        
    Returns:
        Tuple of resampled scanpaths (resampled_sp1, resampled_sp2)
    """
    
    # compute DTW alignment
    distance, path = fastdtw(sp1, sp2, dist=euclidean)
    
    # extract aligned indices
    indices1 = np.array([p[0] for p in path])
    indices2 = np.array([p[1] for p in path])
    
    # create a normalized time axis for the warping path
    path_time = np.linspace(0, 1, len(path))
    
    # create interpolation functions along the warping path
    # scanpath 1
    interp_x1 = interp1d(path_time, sp1[indices1, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_y1 = interp1d(path_time, sp1[indices1, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # scanpath 2
    interp_x2 = interp1d(path_time, sp2[indices2, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_y2 = interp1d(path_time, sp2[indices2, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # generate target times for resampling
    target_times = np.linspace(0, 1, target_length)
    
    # resample both scanpaths
    resampled_sp1 = np.column_stack((interp_x1(target_times), interp_y1(target_times)))
    resampled_sp2 = np.column_stack((interp_x2(target_times), interp_y2(target_times)))
    
    return resampled_sp1, resampled_sp2


# iterate over the simulated data computing the metrics
for value in tqdm(img_ids, desc="Computing metrics"):
    
    # needed in scaled time delay embedding distance
    image_pil = Image.open("../data/images/"+str(value)+".jpg")
    resized_image = image_pil.resize((scale,scale))
    image_np = np.array(resized_image)

    # human scanpaths
    human_scanpath = dR.get_human_scanpath(value, normalized=True, image_width=ref_width, image_height=ref_height)

    #for each human scanpath associated with the image id, calculate the metrics
    for i in range(len(human_scanpath)):
        
        # list of x and y coordinates of the simulated scanpath
        x_list = normalized_df['norm_X'].loc[normalized_df['ID'] == value].values
        y_list = normalized_df['norm_Y'].loc[normalized_df['ID'] == value].values
        simulated_scanpath = np.column_stack((x_list, y_list))
        
        # scaled scanpaths
        simulated_scanpath_scaled=simulated_scanpath*scale
        human_scanpath_scaled=human_scanpath[i]*scale

        # euclidean distance
        dtw_human_scanpath, dtw_simulated_scanpath = resample_scanpath_dtw(human_scanpath[i], simulated_scanpath, min(len(human_scanpath[i]), len(simulated_scanpath)))
        euclidean_distance = sm.euclidean_distance(dtw_human_scanpath, dtw_simulated_scanpath)
        euclidean_distance = np.mean(euclidean_distance)
        euclidean_list.append(euclidean_distance)
        
        # string edit distance
        string_edit_distance_value, _, _ = sm.string_edit_distance(human_scanpath_scaled, simulated_scanpath_scaled, scale,scale,grid_size=100)
        string_edit_distances_list.append(string_edit_distance_value)

        # scanMatch
        scanMatch_value = sm.scanMatch_metric(human_scanpath_scaled, simulated_scanpath_scaled, scale, scale,TempBin=0)
        scanMatch_list.append(scanMatch_value)
        
    
    # mean of metrics 
    mean_euclidean_list = np.mean(euclidean_list)
    mean_string_edit_distances_list = np.mean(string_edit_distances_list)
    mean_scanMatch_list = np.mean(scanMatch_list)


    # sequence core
    x_list = normalized_df['norm_X'].loc[normalized_df['ID'] == value].values.tolist()
    y_list = normalized_df['norm_Y'].loc[normalized_df['ID'] == value].values.tolist()
    
    # creation of the simulated scanpath cluster structure
    prompt_simulated = dR.get_prompt_from_id(value).replace(" ", "_")
    pred_scanpath = {
        'X': x_list,
        'Y': y_list,
        'condition': 'simulated',
        'task': prompt_simulated,
        'name': f'{value}'
    }
    preds = [pred_scanpath]
    key_cluster = f'test-simulated-{prompt_simulated}-{value}'
    
    # iterate over the clusters and compute the sequence score
    if key_cluster in clusters:
        ss_results = sm.compute_SS(preds, {key_cluster: clusters[key_cluster]}, truncate=max_step)
        if ss_results:
            sequence_score = np.mean([res['score'] for res in ss_results])
        else:
            sequence_score = np.nan
    else:
        sequence_score = np.nan
    
    # add the metrics to the dataframe
    new_index = len(metrics_results_df) 
    metrics_results_df.loc[new_index] = [
        value, mean_euclidean_list,
        mean_string_edit_distances_list, 
        mean_scanMatch_list,
        sequence_score
        ]
    
    # lists reset for the next image id
    euclidean_list = []
    string_edit_distances_list = []
    scanMatch_list = []

    
print(f"\nResults:\n\n{metrics_results_df}")
print(f"\n\nMean values:\n\n{metrics_results_df.drop("ID", axis=1).mean(axis=0)}")
