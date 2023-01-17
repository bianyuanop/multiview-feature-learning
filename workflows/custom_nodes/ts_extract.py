from nipype import Node, Function

def get_time_series(template_file, function_image, standarize=True):
    from nilearn.maskers import NiftiLabelsMasker
    masker = NiftiLabelsMasker(labels_img=template_file, standardize=standarize)
    time_series = masker.fit_transform(function_image)
    
    return time_series

ts_extract = Node(Function(input_names=['template_file', 'function_image'], output_names=['time_series'], function=get_time_series), name='ts_extrac')