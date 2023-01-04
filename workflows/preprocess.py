import os
from nipype.interfaces import io as nio  # Data i/o
from nipype.interfaces import spm as spm  # spm
from nipype.interfaces import fsl as fsl  # fsl
from nipype.interfaces import utility as niu  # utility
from nipype.pipeline import engine as pe  # pypeline engine
from nipype.algorithms import misc

def get_preprocess_workflow(
    base_dir,
    out_dir,
    infosource_config = {
        "subject_id": ["08"],
        "run": ["1", "2"], 
    },
    datasource_config = {
        "template_args": {
            'anat': [['subject_id', 'subject_id']],
            'func': [['subject_id', 'subject_id', 'run']]
        },
        "field_template": {
            'anat': 'sub-%s/anat/sub-%s_T1w.nii.gz', 
            'func': 'sub-%s/func/sub-%s_task-flanker_run-%s_bold.nii.gz'
        }
    }
):
    # data_dir = '/home/chan/data/flanker/'
    # subject_list = ["08"]
    # runs = ["1", "2"]

    infosource_iterables = []
    for key in infosource_config:
        infosource_iterables.append((key, infosource_config[key])) 

    infosource = pe.Node(interface=niu.IdentityInterface(fields=list(infosource_config.keys())), name='infosource')
    infosource.iterables = infosource_iterables

    # infosource = pe.Node(interface=niu.IdentityInterface(fields=['subject_id', 'run']), name='infosource')
    # infosource.iterables = [('subject_id', subject_list), ('run', runs)]

    # datasource = pe.Node(
    #     interface=nio.DataGrabber(
    #         infields=['subject_id', 'run'], outfields=['func', 'anat']),
    #     name='datasource')
    datasource = pe.Node(interface=nio.DataGrabber(infields=list(infosource_config.keys()), outfields=list(datasource_config["field_template"].keys())), name='datasource')
    datasource.inputs.base_directory = base_dir
    datasource.inputs.template = '*'
    datasource.inputs.sort_filelist = True

    datasource.inputs.template_args = datasource_config["template_args"]
    # datasource.inputs.field_template = {'func': '_run_%s_subject_id_%s/_fwhm_%s/swarsub-%s_task-flanker_run-%s_bold.nii'}
    datasource.inputs.field_template = datasource_config["field_template"]

    # datasource.inputs.template_args = {
    #     'anat': [['subject_id', 'subject_id']],
    #     'func': [['subject_id', 'subject_id', 'run']]
    # }

    # datasource.inputs.field_template = {
    #     'anat': 'sub-%s/anat/sub-%s_T1w.nii.gz', 
    #     'func': 'sub-%s/func/sub-%s_task-flanker_run-%s_bold.nii.gz'
    # }

    gzFunc = pe.Node(misc.Gunzip(), name='gzFunc')
    gzAnat = pe.Node(misc.Gunzip(), name='gzAnat')

    realign = pe.Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True

    # input: in_files
    # output: timecorrected_files
    timeslice = pe.Node(interface=spm.SliceTiming(), name='timeslicing')
    # timeslice.inputs.in_files = 'functional.nii'
    timeslice.inputs.num_slices = 40
    timeslice.inputs.time_repetition = 2
    # TA = TR - (TR/NUM_SLICES)
    timeslice.inputs.time_acquisition = 2.-(2./40)
    timeslice.inputs.slice_order = [i for i in range(1, 40, 2)] + [i for i in range(2, 41, 2)]
    timeslice.inputs.ref_slice = 1

    # input: 
    #   source: should be anatomical file
    #   target: should be mean func file processed after realignment
    # output: 
    #   coregistered_files
    #   coregistered_source

    coregister = pe.Node(interface=spm.Coregister(), name="coregister")
    # should be estimate and reslice
    coregister.inputs.jobtype = 'estwrite'

    # input: channel_files
    # output: bias_corrected_images, forward_deformation_field
    segment = pe.Node(interface=spm.NewSegment(), name='segment')
    # [Inverse, Forward]
    segment.inputs.write_deformation_fields = [False, True]
    # prob map, gaussian, native tissue, warped tissue
    # ((tissue probability map, frame num), Num gaussians, (Native space, dartel imported), (Unmodulated, Modulated))
    tissue1 = (('/home/chan/tools/spm12/tpm/TPM.nii', 1), 1, (True, False), (False, False))
    tissue2 = (('/home/chan/tools/spm12/tpm/TPM.nii', 2), 1, (True, False), (False, False))
    tissue3 = (('/home/chan/tools/spm12/tpm/TPM.nii', 3), 2, (True, False), (False, False))
    tissue4 = (('/home/chan/tools/spm12/tpm/TPM.nii', 4), 3, (True, False), (False, False))
    tissue5 = (('/home/chan/tools/spm12/tpm/TPM.nii', 5), 4, (True, False), (False, False))
    tissue6 = (('/home/chan/tools/spm12/tpm/TPM.nii', 6), 2, (False, False), (False, False))
    segment.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]

    # input: 
    #   deformation_file: from segment deformation
    #   apply_to_files
    # output:
    #   normalized_files
    #   normalized_image
    normalize = pe.Node(interface=spm.Normalize12(), name="normalize")
    # normalize = interface=spm.Normalize12()
    normalize.inputs.jobtype = 'write'

    smooth = pe.Node(interface=spm.Smooth(), name="smooth")
    fwhmlist = [4]
    smooth.iterables = ('fwhm', fwhmlist)

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = os.path.join(base_dir, out_dir)

    workflow = pe.Workflow(name='datagrab', base_dir='tmp')
    workflow.connect([
        (infosource, datasource, [('subject_id', 'subject_id'), ('run', 'run')]),
        (datasource, gzFunc, [('func', 'in_file')]),
        (datasource, gzAnat, [('anat', 'in_file')]),

        (gzFunc, realign, [('out_file', 'in_files')]),
        (realign, timeslice, [('realigned_files', 'in_files')]),

        (realign, coregister, [('mean_image', 'target')]),
        (gzAnat, coregister, [('out_file', 'source')]),

        (coregister, segment, [('coregistered_source', 'channel_files')]),

        (segment, normalize, [('forward_deformation_field', 'deformation_file')]),
        (timeslice, normalize, [('timecorrected_files', 'apply_to_files')]),

        (normalize, smooth, [('normalized_files', 'in_files')]),

        (smooth, datasink, [('smoothed_files', 'smoothed')]),
    ])

    return workflow

# results = workflow.run()

if __name__ == '__main__':
    workflow = get_preprocess_workflow(
        base_dir=os.getcwd(),
        out_dir='preprocessed',
        infosource_config = {
            "subject_id": ["08"],
            "run": ["1", "2"], 
        },
        datasource_config = {
            "template_args": {
                'anat': [['subject_id', 'subject_id']],
                'func': [['subject_id', 'subject_id', 'run']]
            },
            "field_template": {
                'anat': 'flanker/sub-%s/anat/sub-%s_T1w.nii.gz', 
                'func': 'flanker/sub-%s/func/sub-%s_task-flanker_run-%s_bold.nii.gz'
            }
        })

    workflow.run()