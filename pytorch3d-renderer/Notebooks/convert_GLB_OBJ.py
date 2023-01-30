'''
Script to convert GLB to OBJ.
Using that when trying to solve issue with textures

'''


import requests
import tempfile
from pytorch3d.io import load_obj, load_ply
import sys
sys.path.append("..")
import os
import json
from rendererClass import RendererClass
import itertools
from PIL import Image
import numpy as np
import cv2
sys.path.append("/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/Annotation_API/mv-annotation-service")
import example_client_copy
import matplotlib
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import io
import trimesh

params = {
"image_size": 1024,
"camera_dist": [1.1],  
"elevation": [0,45],
"azim_angle": [0,45,90,135,180,225,270,315,360],
"filename": "",
"z_coord": 0, 
"save_img":False,                        
"save_path": ""
}

params["filename"] = "/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/pytorch3d-renderer/data/random_example/kitchen_v2/kitchen.glb"
Renderer  = RendererClass(params)

Renderer.create_mesh_object()
    
print("The end!")