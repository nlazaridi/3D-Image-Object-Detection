#!/usr/bin/env python
# coding: utf-8

'''
This is the final script prepared for Triton inference
''' 


# Import libraries
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict,Any, Union, Optional
import trimesh
import tempfile
from loguru import logger
import numpy as np
import io

# For installing pytorch3D
'''if torch.__version__ == '1.6.0+cu101' and sys.platform.startswith('linux'):
    
    print("We are here")
    get_ipython().system('pip install pytorch3d')
else:
    need_pytorch3d = False
    try:
        import pytorch3d
    except ModuleNotFoundError:
        need_pytorch3d = True
    if need_pytorch3d:
        get_ipython().system('curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz')
        get_ipython().system('tar xzf 1.10.0.tar.gz')
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        get_ipython().system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")'''
        
        

from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    TexturesUV,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer
)





class RendererClass:
    
    '''
    Class for rendering 3D objects/scenes to multiview images using Pytorch3D
    '''
    def __init__(self,params:Dict[str,Any]) -> None:
        
        """Initializer
        
        Args:
            params (Dict[str,Any]): dictionary with several parameters for initializing the class such as
                                    (filename, save_img,save_path,image_size)
        """
     
        self.filename = params["filename"]
        self.save_img = params["save_img"]
        self.save_path = params["save_path"]
        self.image_size = params["image_size"]
        self.file_extension = params["file_extension"]
        
        #Set the device (GPU or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:3")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu") 



    def pre_process(self, verts):
        
        """Normalize the input to unit sphere
        
        Args:
            verts: 3D vertices of the object/scene
        Returns:
            verts: normalized 3D vertices to unit sphere 
        """ 
        
        
        # ---- normalize and center the input mesh --------
        print("normalize the input in a unit shpere")
        

        # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
        # (scale, center) will be used to bring the predicted mesh to its original center and scale
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        
        return verts


    def load_object(self,exported_path):
        
        """Load the obj object/scene by using buld-in load_obj function from pytorch3D
        

        Args:
            The input comes from the parameters load when initializing the class
            filename: str, path to the 3D  filename
            device: str, the torch device containing a device type ('cpu' or
            'cuda')

        Returns:
            verts, faces, aux: Returns vertices, faces and textures
        """
    
        #CUDA error when using GPU for loading!
        try:
            # Get vertices, faces, and auxiliary information
            verts, faces, aux = load_obj(
                exported_path,
                device="cpu",
                load_textures=True,
                create_texture_atlas=True,
                texture_atlas_size=4,
                texture_wrap="repeat"
                )
        except:
            
            try:
                verts, faces, aux = load_obj(
                    exported_path,
                    device='cpu',
                    load_textures=True
                    )
                
            except RuntimeError as e:
                raise RuntimeError(e)
            
        
        return verts, faces, aux

    def create_mesh_object(self,obj = None):
        
        '''Generates Meshes object for a given mesh with vertices, faces,
        and textures.'''
        
        # Check the format in order to load the object/scene in the correct format
        # For OBJ, directly use pytorch3D load!
        if self.file_extension == "obj":
            
            logger.info(f'Working with OBJ format')
            
            verts,faces,aux = self.load_object(self.filename)
            
            verts = verts.to(self.device)
            faces_idx = faces.verts_idx.to(self.device) #get ids from the faces for back projecting
            
            # normalize and center the mesh
            verts = self.pre_process(verts)
                
            # ------ Create a non-textured object ----------
            mesh = Meshes(
                verts=[verts],
                faces=[faces_idx])
        
        #For all the other formats convert the format to OBJ
        elif self.file_extension in ("off","ply","stl","glb","gltf"):
            
            logger.info(f'Working with not OBJ format')
             
            #load scene with trimesh
            scene = trimesh.load(io.BytesIO(obj),
                                 #force = "scene",
                                 file_type = self.file_extension,
                                 process = False)
            #scene = trimesh.load(self.filename, fix_texture = True, prefer_color = "faces")
            
            #temp directory for exporting OBJ format
            with tempfile.TemporaryDirectory() as tmpdir:

                # The context manager will automatically delete this directory after this section
                logger.info(f"Created a temporary directory: {tmpdir}")
                
                #export any file extension to obj for the renderer
                exported_path_obj = os.path.join(tmpdir,'exported_OBJ.obj')
                
                scene.export(file_type="obj",
                #resolver = trimesh.resolvers.nearby_names(save_path),
                write_texture = True,
                file_obj = exported_path_obj,
                include_color=True,
                include_texture=True)
            
                verts,faces,aux = self.load_object(exported_path_obj)
            
            verts = verts.to(self.device)
            faces_idx = faces.verts_idx.to(self.device) 
            
            # normalize and center the mesh
            verts = self.pre_process(verts)
            
            # ------ Create a non-textured mesh -----------
            mesh = Meshes(
                    verts=[verts],
                    faces=[faces_idx])

        else:
            raise ValueError(f'The {self.file_extension} format is not supported for now')
    
           
        #different options for textures for all formats!
        #atlas texture
        if aux.texture_atlas is not None:   
            
            logger.info(f"Working with atlas texture")
            atlas = aux.texture_atlas.to(self.device)
            
            # Create Meshes object
            mesh.textures = TexturesAtlas(atlas=[atlas])
            
            self.mesh = mesh 
        #if texture not available, generate gray scale tex
        else:
            logger.info(f"Working with colors!")
            
            #OBJ are stored again in the tempfile so we can, this does not hold for all other formats
            if self.file_extension == "obj":
                with open(self.filename, "rb") as object:
                    obj = object.read()
            #load scene with trimesh for getting colors attribute
            scene = trimesh.load(io.BytesIO(obj),
                                #force = "scene",
                                file_type = self.file_extension,
                                process = False)
            
            obj_visual  = getattr(scene, "visual", None)
            
            #if the attribute exists then get the colors
            if obj_visual is not None:
                features = torch.from_numpy(np.array(scene.visual.vertex_colors[:,0:3]/255))[None]
                mesh.textures =  TexturesVertex(verts_features=features.float()).to(self.device)
                logger.info(f"Working with RGB colors")
                
            #else get grayscale colors
            else:
                logger.info(f"Working with random/grayscale texture")
                color = torch.ones(1, mesh.verts_list()[0].shape[0], 3, device=self.device)
                mesh.textures = TexturesVertex(verts_features=color)
                
                
                
                
            self.mesh = mesh 
            
        
    def get_renderer(
        self, dist:int, elev:int, azim:int
        ):
        """
        Generates a mesh renderer by combining a rasterizer and a shader.

        Args:
            self: Gets parameters from initializer
                image_size: int, the size of the rendered .png image
                device: str, the torch device containing a device type ('cpu' or 'GPU')
            dist: int, distance between the camera and 3D object
            elev: int, elevation angle 
            azim: int, azimuth angle

        Returns:
            renderer: MeshRenderer class
        """
        
        # Initialize the camera with camera distance, elevation, azimuth angle,
        # and image size
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        # Initialize rasterizer by using a MeshRasterizer class
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        # The textured phong shader interpolates the texture uv coordinates for
        # each vertex, and samples from a texture image.
        shader = SoftPhongShader(device=self.device, cameras=cameras)
        # Create a mesh renderer by composing a rasterizer and a shader
        renderer = MeshRenderer(rasterizer, shader)
        return renderer


    def render_image(self,dist,elev,azim):
        """
        Renders an image using MeshRenderer class and Meshes object. Saves the
        rendered image as a .png file if save if requested.

        Args:
            self: Gets parameters from initializer
                image_size: int, the size of the rendered .png image
                device: str, the torch device containing a device type ('cpu' or 'GPU')
            dist: int, distance between the camera and 3D object
            elev: int, elevation angle 
            azim: int, azimuth angle

        Returns:
            renderer: MeshRenderer class
        """
        renderer = self.get_renderer(dist,elev,azim)
        image = renderer(self.mesh)
        
        # if true save the image locally
        if self.save_img:
            
            #I suppose this is not need for Triton
            out = os.path.normpath(self.filename).split(os.path.sep)
            mesh_filename = out[-1].split(".")[0]
            dir_to_save =  os.path.join(self.save_path,mesh_filename,"dist_" + str(dist)) 
            os.makedirs(dir_to_save, exist_ok=True)
            sep = '_'
            file_to_save = '{0}{1}{2}{3}{4}{5}{6}{7}'.format(mesh_filename, sep,
                                                            "elev", int(elev),
                                                            sep, "azim",
                                                            int(azim), ".png")
            self.filename_save = os.path.join(dir_to_save, file_to_save)
            print("Saved image as " + str(self.filename_save))
            matplotlib.image.imsave(self.filename_save, image[0, ..., :3].cpu().numpy())
            
            return image[0, ..., :3].cpu().numpy()
        else:
            return image[0, ..., :3].cpu().numpy()



    def combine_all_steps(self,params):
        
        """function to compile all steps and render the 3D scene
            Mostly used for testing purposes
            
        Args:
            params (Dict[str,Any]): dictionary with all parameters need for the renderer
        """        
        
        #load the mesh
        self.create_mesh_object()
        
        all_dist = params["camera_dist"]
        all_elev = params["elevation"]
        all_azim = params["azim_angle"]
        
        
        
        [self.render_image(dist = distance,elev= elev_angle, azim = azim_angle)
            for distance in all_dist for elev_angle in all_elev for azim_angle in all_azim]


    
