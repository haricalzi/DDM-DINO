import json
import numpy as np


class datasetReader(object): 

    def __init__(self,dataset):
        try:
            with open(dataset) as f:
                self.dataset=json.load(f)
        except FileNotFoundError:
            print("Error: file not found")
        except json.JSONDecodeError:
            print("Error: invalid JSON file")

   
    # get the unique REF_IDs from the dataset 
    def get_dataset_image_id(self):

        ref_ids = []
        for element in self.dataset:
            ref_id = element['REF_ID']
            if ref_id not in ref_ids:
                ref_ids.append(ref_id)
        return ref_ids
    
    
    # get the prompt associated with the id
    def get_prompt_from_id(self, id):

        for element in self.dataset:
            if element['REF_ID'] == id: 
                return element['REF_SENTENCE']
    
    
    # get the image associated with the id        
    def get_image_from_id(self, id):
        for element in self.dataset:
            if element['REF_ID'] == id: 
                return element['IMAGEFILE']


    # get humnan scanpaths associated with the id
    def get_human_scanpath(self, id, normalized=False,image_width=None,image_height=None):
       
        human_scanpath_list = []

        for experiment in self.dataset:                           
            if  experiment['REF_ID'] == id and not normalized:
                x_vector=experiment['FIX_X']
                y_vector=experiment['FIX_Y']
                column_matrix = np.column_stack((x_vector,y_vector))
                human_scanpath_list.append(column_matrix)
        
            if experiment['REF_ID'] == id and normalized:
                x_vector=experiment['FIX_X']
                y_vector=experiment['FIX_Y']
                column_matrix = np.column_stack((x_vector,y_vector))
                column_matrix_normalized = column_matrix.copy()
                column_matrix_normalized[:,0] = column_matrix[:,0] / image_width
                column_matrix_normalized[:,1] = column_matrix[:,1] / image_height
                human_scanpath_list.append(column_matrix_normalized)

        return human_scanpath_list
    