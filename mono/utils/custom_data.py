import glob
import os
import json
import cv2
import yaml 
import numpy as np

def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb']
        depth = anno['depth'] if 'depth' in anno else None
        depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
        intrinsic = anno['cam_in'] if 'cam_in' in anno else None
        normal = anno['normal'] if 'normal' in anno else None

        data_i = {
            'rgb': rgb,
            'depth': depth,
            'depth_scale': depth_scale,
            'intrinsic': intrinsic,
            'filename': os.path.basename(rgb),
            'folder': rgb.split('/')[-3],
            'normal': normal
        }
        datas.append(data_i)
    return datas

def load_data(path: str):
    rgbs = glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png')
    #intrinsic =  [835.8179931640625, 835.8179931640625, 961.5419921875, 566.8090209960938] #[721.53769, 721.53769, 609.5593, 172.854]
    data = [{'rgb': i, 'depth': None, 'intrinsic': None, 'filename': os.path.basename(i), 'folder': i.split('/')[-3]} for i in rgbs]
    return data


def yaml_construct_opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node,deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


def read_calib(calib_path):
    yfp = open(calib_path,"r")
    next(yfp,None)
    yaml.add_constructor(u'tag:yaml.org,2002:opencv-matrix', yaml_construct_opencv_matrix,yaml.SafeLoader)
    calib = yaml.safe_load(yfp)
    yfp.close()

    return calib


def load_data_with_calib(path:str, calib_path):
    rgbs = glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png')
    rgbs.sort()

    calib = read_calib(calib_path)
    fx = calib["Camera.fx"]
    fy = calib["Camera.fy"]
    cx = calib["Camera.cx"]
    cy = calib["Camera.cy"]
    intrinsic = [fx, fy, cx, cy] 

    data = [{'rgb': i, 'depth': None, 'intrinsic': intrinsic, 'filename': os.path.basename(i), 'folder': i.split('/')[-3]} for i in rgbs]
    
    return data