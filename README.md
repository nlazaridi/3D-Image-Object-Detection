**A repository for 3D rendering**


This doc provides documentation for the 3D annotation code with multiview images. the idea is simple, renderer the 3D object/scene with several images. Then pass each image to the 2D object detection model, aggregate the predictions and get a list of the unique values of them.

**You only need to use the following scripts**: rendererClass.py, rendering.py and multiview_annotation.py


# Install libraries

The entire pipeline is based in pytorch3D, which also needs torch

In the script rendererClass.py there is a chunk of code to install pytorch3D

Specifically, for torch 1.10 (which is the version of torch in the annotation API if I am not mistaken), the new released 0.6.1 version is built with wheel.

**Other libraries that you may need to install**: os, typing, matplotlib, sys, argparse, json, trimesh

You may use the environment.yml to create the conda environment (it also contains the libraries needed to run the annotation client). But it may have more libraries than needed in the final pipeline

# Data 

There is not a specific dataset that can be used. We mostly search for different 3D models online and download them (check the [doc](https://docs.google.com/document/d/1CY1QLmUPjZYABDi9k6thm0s6Iih1SPLknXpsprw9_8M/edit)). Download the data, and put them in the root dir. You can visualize them with Meshlab or cloudcompare. 

Several data examples can be found in the copy of the hard drive in the data folder.

# How to run

You can use the script 3D_multiview_annotation.py to directly render a 3D object/scene

The client (annotation API is in another [repository](https://github.com/gpan12/mv-annotation-service)). You should discuss with Panagiotis to give you access. You can then clone this repo locally and use the example client from there. 

First define the parameters in params_inference.json

python 3D_multiview_annotation.py --pars_path --client_path --use_img

It can work for both URL from drive or local paths



# Multiview pipeline


## Class for rendering (rendererClass.py)

We have create a class for rendering, which can be used to render the 3D scene. It can work for several formats such as obj, ply, gltf, stl, off with limitations in some formats though.

The main renderer is working with OBJ. So for other formats you need to convert them to OBJ. This is a limitation of that approach. We tried other renderers but the one from pytorch seems to be the best one.

In order to use the class for rendering, several parameters are needed. These parameters are passed in the class as a json (see params_inference.json for more details on the parameters).


**Example of params_inference.json:**

{
"image_size": 1024,

"camera_dist": [2],  

"elevation": [0],

"azim_angle": [0,45,90,135,180,225,270,315,360],

"filename": "/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/pytorch3d-renderer/data/random_example/Police_Officer/UrbanPoliceOfficer.obj",

"file_extension": "obj",

"z_coord": 0, 

"save_img":true,                        

"save_path": "/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/pytorch3d-renderer/results/"
}


* `image_size` is a size of an actual 2D output image. The smaller the size, the more pixelated the image will appear. Try 512 or 1024 to get crisp images but, by the same token, the code will take longer to run.\
* `camera_dist` refers to the distance between the camera and the object.\
* `elevation` is a **list** of elevation values and basically tell us from how high we are looking at the object. **Elevation refers to the angle between the vector from the object to the camera and the horizontal plane y=0 (plane xz).**\
* `azim_angle` is a **list** of azimuth angle values and basically tell us from which side (e.g. left size, right side, front view, back view, etc.) we are looking at the object. What's azimuth angle? **Let's say you have a vector from the object to the camera and you project it onto a horizontal plane y=0. The azimuth angle is then the angle between the projected vector and a reference vector at (0,0,1) on the reference plane (horizontal plane).** [Checkout](https://www.celestis.com/resources/faq/what-are-the-azimuth-and-elevation-of-a-satellite/) this illustration.\
* `filename` is a path to the file you want to render (So far only .obj is supported). Note: You need to specify the specific file not the entire folder. For example UrbanPoliceOfficer.obj 
* `file_extension` is the format of the 3D object (can be obj, ply, off, glb, gltf or stl for now)
* `save_img`: if True the images are locally stored otherwise just return the image
* `save_path`: path to store the images if save_img = true

So far the parameters above appear to be the optimal solution for both objects and scenes.



## Step-by-step Pipeline

**Rendering in one step**

As mentioned earlier you first need to get access in the client's repo. 

After the client was updated, you can use the script 3D_multiview_annotation.py to directly render a 3D object/scene

First define the parameters in params_inference

python 3D_multiview_annotation.py --pars_path --client_path --use_img

It can work for both URL from drive or local paths

The script will return a list of the annotations.



**Rendering in two steps:**

This way was initially used for first rendering the 3D object/scene, then upload the png to Drive and use the annotator as last step. We mostly use that in order to check visually the output of the renderer.

* For rendering a specific 3D object/scene, you just need to modify the params_inference.json and run the script: python rendering.py --params_path --renderer_path

    The images are then store in the save_path (which is defined in the params_inference.json). Then it needs to be uploaded somewhere in order to be downloaded with a link (for instance Gdrive)

* Then use the client for annotating a list of images with: python multiview_annotation.py --URLs --client_path

    For a given string of urls in order to get back the list of the annotations in the scene.

    As input, you need to give a string of all the urls (this is used because you can extract this link directly from Gdrive) and the path where the client is located

    The output is a list of all the annotations. So far we just merge the annotations and get the unique annotations.



## developments

The developer I was talking to is Panagiotis Galopoulos (github gpan12). You can ask him for any questions regarding the deployment of the models. We had a group in skype for communication.

**Deployment step-by-step**

* Start with this repo in order to update the renderer, conduct experiments etc. I was using  API_deployment.ipynb in the Notebooks folder for testing mostly.

* When there is something major push the changes in this [repo](https://github.com/mever-team/media-annotation-3d-service) (See below for more details)

* Thn contact Panagiotis to check if the API is working properly

So far the renderer works with obj, ply, off, glb, gltf or stl


### Documentation of the annotation [repo](https://github.com/mever-team/media-annotation-3d-service)

When you have a major update you push your code in this repo. The repo is dockerized and ready to be used.

You most probably need to change only the two scipts (annotate_3d.py and rendererClass.py) in media-annotation-3d-service/app/endpoints/.

The rendererClass.py is the same as the one rendereClass.py in this repo. For example, if you add support for more formats, you can update renderClass without changing anything else in the repo. 

The annotate_3d.py is the main script that runs the renderer. It structure is very similar to our 3D_multiview_annotation.py

So you mostly work with 3D_multiview_annotation.py and rendererClass.py. When you have updates to pass them in the repo in the rendereClass.py and annotate_3d.py.


In order to test if there is no bug, you need to compile the docker image. To do that you can run in the root directory of the media-annotation-3d-service repo: 

* docker build . -t 3d-annotate-test

* docker run --gpus all -p 8080:80 3d-annotate-test

* Then check it. For example, if the server is 160.40.52.109:8080/docs 


If you have any issues, you can contact Panagiotis to help you with the docker etc.



# Folder description

There are several folders used in the repo

* data: Download the data from here and add it to the root dir

* Notebooks: All the notebooks used during developments. Most of the notebooks were used to convert different formats in OBJ. I provide a small description in each one of the notebooks.

* Utils: Some util functions (mostly not used)

* 3D_multiview_annotation.py: This is the main script to run for 3D annotation

* multiview_annotation_from_URL.py: Initial script used for annotation URL objects. Probably not need to use it anymore

* params_inference: Define the parameters of the renderer

* rendererClass: The class of the renderer to use

* rendering.py: A script to render an object only (so no annotation client is used)


# Improvements/Issues

In general the renderer is in a quite immature stage. There are a lot of things to consider in order to make it robust.

* From my discussion with Stephan/Dimitris the 3D objects will mostly be in glb/gltf format (at least the ones from the immersive tools). The **huge** issue is the textures from the GLB to OBJ format. This happens **ONLY** when there are several geometries. In practice, if you convert a glb with several textures to obj, only one texture is exported. This is an issue with export_obj from Trimesh library (same thing happens if you export Obj to Obj). On the other hand, if the object has only one texture it works perfectly. We have tried the following solutions:

    * Tried to convert from GLB with trimesh, load directly to pytorch3D, store each mesh seperately etc --> Nothing gave a better result!
    * Better to wait for  answer in the [issue](https://github.com/mikedh/trimesh/issues/1499) and check again what can we do (FOR NOW FROZE IT). I have created an issue in Trimesh library and it seems that there are other people with the same problem.
    * Alternatives :Different RENDERER (Pyrender,Pygltflib,gltflib for instance). I have tried some of them but it needs a lot of effort to change renderer.


* An issue right now with the annotation service is that the user can only upload one file. On the other hand OBJ format (and some other formats) may need several files. For example, obj need 3 files (texture, mtl and the actual obj). There was a discussion with ATC to provide use with a zip (when the user uploads several files). If that is the case, you can discuss with Panagiotis how to proceed. 

* Fix orientation issue (mostly for PLY and STL). Specifically, PLY most times has different orientation from the OBJ and therefore the images are not correctly rendered


* Regarding the code in the [repo](https://github.com/mever-team/media-annotation-3d-service), I think it is not very efficient. There are several things that can be done but I did not have time to improve them. 

    * A lot of optimizations that can be done. For instance, now the rendered images are stored in a temp file and then used as url to the annotation client. What can be done is to directly send the image to the client in bytes format (I have implement it in the 3D_multiview_annotation.py in this repo, but not deployed it to the API)

* Another important issue is how to treat the difference between object and scene. 

    * When you use the pipeline for an object as it is now, it may return more than one objects. For example, the object is a desk and the service may return desk and table.
    
    
    * Additionally, the distance from the camera should be different when we have an object and when we have a scene. From my experiments I saw that for object dist = 2 and for scenes dist = 1.1 is the optimal params. 

    * One solution, would be to ask from ATC to provide you with the type. For example object or scene. Then you can treat them differently. For instance if it is an object, return only the majority class. 



* I do not think there is a need for supporting more formats, but it is better to follow the meetings and see if other formats will be used.

    


In summary, the renderer is quite immature. A lot of staff are needed to be done to be robust. I think a good starting point is to run some examples, familiarize yourself with the task and then discuss with Spyros if it worths to continue with that or you can focus in something else.



**Rescources**


* Pytorch3D tutorials: https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb, https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_textured_meshes.ipynb 