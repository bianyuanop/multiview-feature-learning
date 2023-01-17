import os
from nipype.interfaces import io as nio  # Data i/o
from nipype.interfaces import spm as spm  # spm
# from nipype.interfaces import matlab as mlab    # how to run matlab
from nipype.interfaces import fsl as fsl  # fsl
from nipype.interfaces import utility as niu  # utility
from nipype.pipeline import engine as pe  # pypeline engine

def get_multi_atlas_workflow(
    base_dir, 
    fsl_dir="fsl",
    infosource_config = {
        "subject_id": ["08"],
        "run": ["1", "2"], 
        "fwhm": ["4"]
    },
    datasource_config = {
        "template_args": {
            'func': [['run', 'subject_id', 'fwhm', 'subject_id', 'run']]
        },
        "field_template": {
            'func': 'processed/smoothed/_run_%s_subject_id_%s/_fwhm_%s/swarsub-%s_task-flanker_run-%s_bold.nii'
        }
    }
):

    # base_dir = os.getcwd()
    fsl_dir = os.path.join(base_dir, fsl_dir)

    aal = os.path.join(fsl_dir, 'aal/AAL3v1.nii')
    target = os.path.join(fsl_dir, 'std/MNI152_T1_2mm_brain.nii')

    # info
    # data_dir = os.path.join(base_dir, 'processed/smoothed/')
    # subject_list = subject_list
    # runs = ["1", "2"]
    # fwhm_list = ["4"]
    infosource_iterables = []
    for key in infosource_config:
        infosource_iterables.append((key, infosource_config[key])) 

    infosource = pe.Node(interface=niu.IdentityInterface(fields=list(infosource_config.keys())), name='infosource')
    infosource.iterables = infosource_iterables

    # data source
    # datasource = pe.Node(interface=nio.DataGrabber( infields=['subject_id', 'run', 'fwhm'], outfields=['func']), name='datasource')
    datasource = pe.Node(interface=nio.DataGrabber(infields=list(infosource_config.keys()), outfields=list(datasource_config["field_template"].keys())), name='datasource')

    datasource.inputs.base_directory = base_dir
    datasource.inputs.template = '*'
    datasource.inputs.sort_filelist = True
    # datasource.inputs.template_args = {
    #     'func': [['run', 'subject_id', 'fwhm', 'subject_id', 'run']]
    # }

    datasource.inputs.template_args = datasource_config["template_args"]
    # datasource.inputs.field_template = {'func': '_run_%s_subject_id_%s/_fwhm_%s/swarsub-%s_task-flanker_run-%s_bold.nii'}
    datasource.inputs.field_template = datasource_config["field_template"]

    flirt_aff = pe.Node(interface=fsl.FLIRT(), name='flirt_aff')
    flirt_aff.inputs.reference = os.path.join(base_dir, 'fsl/std/MNI152_T1_2mm_brain.nii')

    convert_xfm = pe.Node(interface=fsl.ConvertXFM(), name='inverse_aff')

    # input: in_file, config_file 
    # output: fieldcoeff_file
    fnirt = pe.Node(interface=fsl.FNIRT(), name='fnirt')
    fnirt.inputs.ref_file = os.path.join(base_dir, 'fsl/std/MNI152_T1_2mm_brain.nii')
    fnirt.inputs.fieldcoeff_file = True
    fnirt.inputs.warped_file = 'warped.nii.gz'

    # take name of a file in target space and warp coefficient
    # A -> B: AwarpB
    # inverse should take AwarpB and A since target space is A
    # input: reference, warp
    # output: inverse_wrap
    inv_warp = pe.Node(interface=fsl.InvWarp(), name='inv_warp')

    flirt_aal = pe.Node(interface=fsl.FLIRT(), name='flirt_aal')
    flirt_aal.inputs.in_file = os.path.join(base_dir, 'fsl/aal/AAL3v1.nii')
    flirt_aal.inputs.out_file = 'aalFlirted.nii.gz'

    # input: in_file, ref_file, field_file
    # output: out_file
    apply_warp = pe.Node(interface=fsl.ApplyWarp(), name='apply_warp')
    # apply_warp.inputs.in_file = os.path.join(base_dir, 'fsl/aal/AAL_warped2MNI.nii');

    # used to restore flirt_aff
    flirt_restore = pe.Node(interface=fsl.FLIRT(), name='flirt_restore')

    # linear register non-linear registered atlas to MNI
    flirt = pe.Node(interface=fsl.FLIRT(), name='flirt')
    flirt.inputs.reference = os.path.join(base_dir, 'fsl/std/MNI152_T1_2mm_brain.nii')

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = os.path.join(base_dir, 'inverted')

    workflow = pe.Workflow(name='inv_nl_register', base_dir='multi_atlas')
    workflow.connect([
        (infosource, datasource, [('subject_id', 'subject_id'), ('run', 'run'), ('fwhm', 'fwhm')]),
        # (datasource, gzFunc, [('func', 'in_file')]),

        (datasource, flirt_aff, [('func', 'in_file')]),

        (flirt_aff, convert_xfm, [('out_matrix_file', 'in_file')]),

        (flirt_aff, fnirt, [('out_matrix_file', 'affine_file')]),
        (datasource, fnirt, [('func', 'in_file')]),

        (fnirt, flirt_aal, [('warped_file', 'reference')]),

        (fnirt, inv_warp, [('fieldcoeff_file', 'warp')]),
        (datasource, inv_warp, [('func', 'reference')]),

        (datasource, apply_warp, [('func', 'ref_file')]),
        (inv_warp, apply_warp, [('inverse_warp', 'field_file')]),
        (flirt_aal, apply_warp, [('out_file', 'in_file')]),

        (apply_warp, flirt_restore, [('out_file', 'in_file')]),
        (convert_xfm, flirt_restore, [('out_file', 'in_matrix_file')]),
        (datasource, flirt_restore, [('func', 'reference')]),

        (flirt_restore, flirt, [('out_file', 'in_file')]),

        (flirt, datasink, [('out_file', 'atlas')]),
        (flirt, datasink, [('out_matrix_file', 'atlas_mat')]),
    ])

    return workflow
    # workflow.write_graph(graph2use='flat')

if __name__ == '__main__':
    workflow = get_multi_atlas_workflow(
        base_dir='/home/chan/Documents/graduate-essay', 
        fsl_dir='fsl', 
        infosource_config = {
            "subject_id": ["08", "09", "10", "11"],
            "run": ["1"], 
            "fwhm": ["4"]
        },
        datasource_config = {
            "template_args": {
                'func': [['run', 'subject_id', 'fwhm', 'subject_id', 'run']]
            },
            "field_template": {
                'func': 'preprocessed/smoothed/_run_%s_subject_id_%s/_fwhm_%s/swarsub-%s_task-flanker_run-%s_bold.nii'
            }
        })
    
    workflow.run()
