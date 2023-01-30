'''
This is the main script for rendering and annotating 3D objects/scenes



'''

# import libraries
from ast import arg
import sys
import argparse
import glob
import itertools
import json
from typing import List
sys.path.append("..")
from rendererClass import RendererClass
import time
import shutil
from PIL import Image
import io
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from loguru import logger


# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--pars_path', 
                    type = str,
                    default = "/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/pytorch3d-renderer/params_inference.json",
                    help = "The path were all neede parameters are stored")

parser.add_argument('--client_path',
                    default = "/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/Annotation_API/mv-annotation-service",
                    help = "Path where the annotation client is stored")

parser.add_argument('--use_img',
                    action = "store_true",
                    help = "If True the images is directly used, else it is loaded from local folder")

args = parser.parse_args()


#append path to import the client

''' 
In order for that to work you need to have access to the client repo (ask Panagiotis) 
Then you can clone that repo and send the path in order to import the client
'''

sys.path.append(args.client_path)
import example_client_copy


pars_path = args.pars_path
use_img = args.use_img

#  initialize client
client = example_client_copy.MVAnnotationClient(
        address="160.40.53.61:37527",
        secure=False,
    )




start_time = time.time()

#main function 
def multiview_annotation(param_path, # path where the json is stored
                         use_img # if True the image is directly used in bytes else it is stored locally 
                         ):
    
    annotations = [] #List to store the final annotations
      
     
    #load the parameters 
    with open(param_path) as f:
        params = json.load(f)
    
    print(f'Working with local path, load from {params["filename"]}')
    
    
    # --------- save and load the obj as it is done in the API --------
    '''
    The code below is not needed in general
    We use it to have a better simulation of how it runs in the server
    '''
    
    # read the object 
    with open(params["filename"], "rb") as object:
        obj = object.read()
    
    
    temp_path =  os.path.normpath(params["filename"])
    src_dir = os.sep + os.path.join(*temp_path.split(os.sep)[:-1])
    
    file_extension = params["file_extension"]
    print(f'File is written to: {src_dir}')
    
    ''' 
    if the type of 3D object is obj, you write it locally in order to import it with pytorch3D library
    Working directly with bytes causes errors
    
    '''
    if file_extension == "obj":
        with open(os.path.join(src_dir,"object." + file_extension),"wb") as f:
            f.write(obj)
            
            params["filename"] = f.name
        
    
    # -----------------------------------------------------------------
    
    #initialize the renderer
    Renderer  = RendererClass(params)

    #Load the file and create a mesh object
    #Filename is in the params
    Renderer.create_mesh_object(obj)
    
    #load the pars to render 3D
    all_dist = params["camera_dist"]
    all_elev = params["elevation"]
    all_azim = params["azim_angle"]
    
    #get all combinations of the parameters 
    all_combs =  list(itertools.product(*[all_dist,all_elev,all_azim]))

    print(f'Number of images to annotate: {len(all_combs)}')

    #for loop to go through all the provided multiview images
    for comb in all_combs:
        
        print(f'Working with combination:{comb}')
        
        
        
        #store the image
        img = Renderer.render_image(comb[0],comb[1],comb[2])
        
        
        # manually store the image for QA
        out = os.path.normpath(temp_path).split(os.path.sep)
        mesh_filename = out[-1].split(".")[0]
        dir_to_save =  os.path.join(params["save_path"],mesh_filename,"dist_" + str(comb[0])) 
        os.makedirs(dir_to_save, exist_ok=True)
        sep = '_'
        file_to_save = '{0}{1}{2}{3}{4}{5}{6}{7}'.format(mesh_filename, sep,
                                                        "elev", int(comb[1]),
                                                        sep, "azim",
                                                        int(comb[2]), ".png")
        
        
        filename_save = os.path.join(dir_to_save, file_to_save)
        matplotlib.image.imsave(filename_save, img)

        if use_img:
            
            print(f'Work with img directly:')
            
            #convert numpy image to bytes
            PIL_im = Image.fromarray((img * 255).astype(np.uint8))
            img_byte_arr = io.BytesIO()
            PIL_im.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            
            # Use the annotation client
            try:
                resp = client.sync_annotate(
                    asset_url=None,
                    asset_path=None,
                    asset_bytes = img_byte_arr,
                    models = [
                        example_client_copy.AnnotationModel.OBJECT_DETECTION,
                    ],
                    metadata={"id": "1"},
                )
                object_detection_result, error = resp.get_result(
                    example_client_copy.AnnotationModel.OBJECT_DETECTION
                )

                if object_detection_result:
                    
                    # check is the specific model result is ok
                    print(object_detection_result)
                    
            except example_client_copy.AnnotationError:
                # Something pretty bad happened and all responses are lost.
                # This error should not happen under normal circumstances.
                # It's probably a bug that needs to be reported.
                print("An error occurred")
        
        
        
        else:
            
            #define the asset path of the png
            asset_path = Renderer.filename_save
            print(f'Asset path: {asset_path}')
            
            # Use the annotation client
            try:
                resp = client.sync_annotate(
                    asset_url=None,
                    asset_path=asset_path,
                    models = [
                        example_client_copy.AnnotationModel.OBJECT_DETECTION,
                    ],
                    metadata={"id": "1"},
                )
                object_detection_result, error = resp.get_result(
                    example_client_copy.AnnotationModel.OBJECT_DETECTION
                )

                if object_detection_result:
                    
                    # check is the specific model result is ok
                    print(object_detection_result)
                    
            except example_client_copy.AnnotationError:
                # Something pretty bad happened and all responses are lost.
                # This error should not happen under normal circumstances.
                # It's probably a bug that needs to be reported.
                print("An error occurred")
        
        #append the annotation list with new detections
        [annotations.append(annot.object) for annot in object_detection_result.detections]


    return set(annotations)





if __name__ == "__main__":
    
    annotations = multiview_annotation(pars_path,use_img)
    print(annotations)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Annotation part is Done')


