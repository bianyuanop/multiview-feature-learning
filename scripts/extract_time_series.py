import os
import re
import numpy as np
import typer
from nilearn.maskers import NiftiLabelsMasker

def remove_base(base_dir: str, target: str):
    if not (base_dir in target):
        return '' 
    
    res = target[len(base_dir):]
    if res.startswith('/'):
        res = res[1:]
    
    return res

def get_file_list_by_pattern(start_dir: str, pattern: str):
    files = []
    for r, d, f in os.walk(start_dir):
        for file in f:
            if re.match(pattern, file):
                files.append(os.path.join(r, file))

    return files

def get_relative_folder(filename: str):
    pos = filename.rfind('/')
    return filename[:pos]

def remove_tail(tail: str, target: str):
    if not (tail in target):
        return target
    
    return target[:-len(tail)]
        

def convert2timeseries(template_filename: str, functional_image: str, standarize=True):
    masker = NiftiLabelsMasker(labels_img=template_filename, standardize=standarize)
    time_series = masker.fit_transform(functional_image)
    
    return time_series

def convert2timeseires_batch(start_dir: str, target_pattern: str, template_filename: str, output_dir='output', standarize=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    files = get_file_list_by_pattern(start_dir=start_dir, pattern=target_pattern)
    for filename in files:
        ts = convert2timeseries(template_filename=template_filename, functional_image=filename, standarize=standarize)
        output_filename = remove_base(start_dir, filename)
        output_filename = remove_tail('.nii.gz', output_filename) + '.npy'
        output_fullpath = os.path.join(output_dir, output_filename)
        dir2store = get_relative_folder(output_fullpath)

        print(f'converting {filename} to timeseries at {output_fullpath}')

        if not os.path.exists(dir2store):
            os.makedirs(dir2store)
        
        with open(output_fullpath, 'wb+') as f:
            np.save(f, ts)
        
        

if __name__ == '__main__':
    # start_dir = '/home/chan/data/RawData/Funimg'
    # template_filename = '/home/chan/Documents/graduate-essay/fsl/aal/AAL3v1.nii'
    # target_pattern = 'merged.nii.gz'

    # convert2timeseires_batch(start_dir=start_dir, template_filename=template_filename, target_pattern=target_pattern)
    # pass

    typer.run(convert2timeseires_batch)