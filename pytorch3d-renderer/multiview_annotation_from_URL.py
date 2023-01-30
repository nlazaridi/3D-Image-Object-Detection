import sys
import argparse
import glob

from typing import List

parser = argparse.ArgumentParser()
parser.add_argument('--URLs', 
                    type = str,
                    help = "A string of all the images' urls from Gdrive")

parser.add_argument('--client_path',
                    default = "/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/Annotation_API/mv-annotation-service",
                    help = "Path where the annotation client is stored")


args = parser.parse_args()


#append path to import the client
sys.path.append(args.client_path)
import example_client_mine

# --------------------------- Convert the URLs for being able to be downloaded -------------------------------------
'''
So far we use Gdrive to store the URLs so we need to convert them from view to download format
'''



#all_links_list = glob.glob("/home/andstasi/Projects/MediaVerse/3D_to_2D_converter/pytorch3d-renderer/results/UrbanPoliceOfficer/dist_2/*")


#prefix for downloading a url from Gdrive 


#get all links
all_links_list = args.URLs.split(",") #get all url as list
download_prefix = 'https://docs.google.com/uc?export=download&id='

download_url = []

#convert all urls from view to download url
for link in all_links_list:
    download_url.append(download_prefix+link.strip().split("/")[-2])
    
    
print(f'Number of files to annotate: {len(all_links_list)}')


#  initialize client
client = example_client_mine.MVAnnotationClient(
        address="160.40.53.61:37527",
        secure=False,
    )



def multiview_annot_from_url(URLs:List[str])-> List[str]:
    
    """Returns unique annotations using the annotation client for a list of multiview images 

    Args:
        URLs (List[str]): A string of all URLs (separated by comma) of all images stored in a Gdrive folder


    Returns:
        (List[str]): A list of unique annotations
    """    ''''''
    annotations = [] #List to store the final annotations

    #for loop to go through all the provided multiview images
    for URL in URLs:
        print(URL)
        try:
            resp = client.sync_annotate(
                asset_url=URL,
                asset_path=None,
                models = [
                    example_client_mine.AnnotationModel.OBJECT_DETECTION,
                ],
                metadata={"id": "1"},
            )
            object_detection_result, error = resp.get_result(
                example_client_mine.AnnotationModel.OBJECT_DETECTION
            )

            if object_detection_result:
                
                # check is the specific model result is ok
                print(object_detection_result)
                
        except example_client_mine.AnnotationError:
            # Something pretty bad happened and all responses are lost.
            # This error should not happen under normal circumstances.
            # It's probably a bug that needs to be reported.
            print("An error occurred")
        
        #append the annotation list with new detections
        [annotations.append(annot.object) for annot in object_detection_result.detections]


    return set(annotations)





if __name__ == "__main__":
    annotations = multiview_annot_from_url(download_url)
    print(annotations)


