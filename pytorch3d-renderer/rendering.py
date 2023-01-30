'''
Scipt using the RendererClass for rendering 3D object/scenes to multiview images
All details needed are stored in the parameters json file
'''

import sys
import json
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--params_path', 
                    default = "/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/pytorch3d-renderer/params_inference.json",
                    help = "Path where the json file with the needed parameters is stored")

parser.add_argument('--renderer_path',
                    default = "/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/pytorch3d-renderer/",
                    help = "Path where the RendererClass is stored")

args = parser.parse_args()

#import the rendererClass
sys.path.append(args.renderer_path)
from rendererClass import RendererClass

 
#load the parameters
with open(args.params_path) as f:
    params = json.load(f)

#Initialize the renderer
Renderer  = RendererClass(params)

# Combine all steps to render the 3D object/scene to multiview images
Renderer.combine_all_steps(params=params)