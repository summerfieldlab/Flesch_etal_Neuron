function pipeline_preprocess_fmri()
    %% PIPELINE_PREPROCESS_FMRI()
    %
    % performs preprocessing of raw fMRI data
    %
    
    
    params = fmri_preproc_setParams();
  
    addpath(params.dir.spmDir);
  
    %% DICOM Import
    if params.flags.dicom2Nifti
      fmri_preproc_dicomImport();
    end
  
    %% Realignment
    if params.flags.doRealign
      fmri_preproc_realignUnwarp();
    end
  
    %% Slice Time Correction
    if params.flags.doSTCorr
      fmri_preproc_slicetimeCorr();
    end
  
    %%  Coregistration
    if params.flags.doCoreg
      fmri_preproc_coregistration();
    end
  
    %% Tissue segmentation
    if params.flags.doSegment
      fmri_preproc_segmentation();
    end
  
    %% Normalisation
    if params.flags.doNorm
      fmri_preproc_normalisation();
    end
  
    %% Smoothing
    if params.flags.doSmooth
      fmri_preproc_smooth();
    end
  