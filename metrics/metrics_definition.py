import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance


''' created: Dario Zanca, July 2017

    Implementation of the Euclidean distance between two scanpath of the same length. '''


'''Euclidean Distance'''

def euclidean_distance(human_scanpath, simulated_scanpath):
    if len(human_scanpath) == len(simulated_scanpath):

        dist = np.zeros(len(human_scanpath))
        for i in range(len(human_scanpath)):
            P = human_scanpath[i]
            Q = simulated_scanpath[i]
            dist[i] = np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)
        return dist

    else:

        print('Error: The two sequences must have the same length!')
        return False

###############################################################################################################################################

''' created: Dario Zanca, July 2017

    Implementation of the string edit distance metric.
    
    Given an image, it is divided in nxn regions. To each region, a letter is assigned. 
    For each scanpath, the correspondent letter is assigned to each fixation, depending 
    the region in which such fixation falls. So that each scanpath is associated to a 
    string. 
    
    Distance between the two generated string is then compared as described in 
    "speech and language processing", Jurafsky, Martin. Cap. 3, par. 11. '''

'''String edit distance'''

def scanpath_to_string(
        scanpath,
        stimulus_width,
        stimulus_height,
        grid_size
):
    width_step = stimulus_width // grid_size
    height_step = stimulus_height // grid_size

    string = ''

    for i in range(np.shape(scanpath)[0]):
        fixation = scanpath[i].astype(np.int32)
        correspondent_square = (fixation[0] // width_step) + (fixation[1] // height_step) * grid_size
        string += chr(97 + int(correspondent_square))

    return string

def string_edit_distance(
        scanpath_1,
        scanpath_2,
        stimulus_width,
        stimulus_height,
        grid_size=4
):

    string_1 = scanpath_to_string(
        scanpath_1,
        stimulus_width,
        stimulus_height,
        grid_size
    )
    string_2 = scanpath_to_string(
        scanpath_2,
        stimulus_width,
        stimulus_height,
        grid_size
    )

    return edit_distance(string_1, string_2, transpositions=True), string_1, string_2

###############################################################################################################################################

''' ScanMatch. Evaluating simirality between two fixation sequences with ScanMatch algorithm,
    proposed by Cristino, Mathot, Theeuwes and Gilchrist (2010). 
    Builds upon the GazeParser python package (https://pypi.org/project/GazeParser/)'''

'''ScanMatch'''

from _scanMatch import ScanMatch


def scanMatch_metric(s1, s2, stimulus_width, stimulus_height, Xbin=14, Ybin=8, TempBin=50):

    if s1.shape[1] > 2 and s2.shape[1] > 2:
        tb = TempBin
        diff1 = (s1[:,3] - s1[:,2]) * 1000.
        s1 = np.hstack([s1[:,0].reshape(-1,1), s1[:,1].reshape(-1,1), diff1.reshape(-1,1)])
        diff2 = (s2[:,3] - s2[:,2]) * 1000.
        s2 = np.hstack([s2[:,0].reshape(-1,1), s2[:,1].reshape(-1,1), diff2.reshape(-1,1)])

    elif s1.shape[1] == 2 and s2.shape[1] == 2:
        tb = 0.0
    else:
        raise ValueError("Scanpaths should be arrays of shape [nfix,2] or [nfix,4]! Columns are assumed to be x,y fixations coordinates and, eventually, their start and end times (seconds)")
    
    matchObject = ScanMatch(Xres=stimulus_width, Yres=stimulus_height, Xbin=Xbin, Ybin=Ybin, TempBin=TempBin)
    seq1 = matchObject.fixationToSequence(s1).astype(int)
    seq2 = matchObject.fixationToSequence(s2).astype(int)

    (score, _, _) = matchObject.match(seq1, seq2)

    return score

#############################################################################################################

"""
  @InProceedings{Mondal_2023_CVPR,
    author    = {Mondal, Sounak and Yang, Zhibo and Ahn, Seoyoung and Samaras, Dimitris and Zelinsky, Gregory and Hoai, Minh},
    title     = {Gazeformer: Scalable, Effective and Fast Prediction of Goal-Directed Human Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {1441-1450}}
"""

'''Sequence Score'''

from tqdm import tqdm

def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0

def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))
    

def scanpath2clusters(meanshift, scanpath):
    string = []
    xs = scanpath['X']
    ys = scanpath['Y']
    for i in range(len(xs)):
        symbol = meanshift.predict([[xs[i], ys[i]]])[0]
        string.append(symbol)
    return string

def compute_SS(preds, clusters, truncate, reduce='mean', print_clusters = False):
    results = []
    for scanpath in preds:
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                        scanpath['name'])
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for gt in strings.values():
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)
                score = nw_matching(pred, gt)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results

def get_seq_score(preds, clusters, max_step, tasks=None, print_clusters = False):
    results = compute_SS(preds, clusters, truncate=max_step, print_clusters=print_clusters)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))
    
#############################################################################################################