function params = fmri_preproc_setParams()
%% fmri_preproc_setParams
%
% sets all parameters for preprocessing of fmri data
%
% Timo Flesch, 2019

  params = struct();

  %% Flags
  params.flags.doRealign  =  true;
  params.flags.doSTCorr   =  true;
  params.flags.doCoreg    =  true;
  params.flags.doNorm     =  true;
  params.flags.doSmooth   =  true;
  params.flags.doSegment  = false;

  %% Directories
  params.dir.spmDir = '';

  params.dir.imDir        = 'project/data/data_fmri/renamed/';
  params.dir.conditionDir = 'project/data/data_behav/final/fmri_identifiers/scan/conditions/';
  params.dir.runSubDir    =        'ep2d_64mx_35mm_TE30ms_';
  % params.dir.epiSubDir    = 'functional/';
  params.dir.structSubDir = 't1_mpr_sag_p2_iso_0004/';
  params.dir.estSubDir    =  'estimates/';

  %% File Names
  % regular expressions for filenames
  params.regex.epiNormImages     = '^wauf.*\.nii$'; % functional images
  params.regex.epiCoregImages    = '^auf.*\.nii$'; % functional images
  params.regex.structRawImages   =   '^s.*\.nii$';   % structural images (coregistered)
  params.regex.structCoregImages =  '^rs.*\.nii$';  % structural images (coregistered)
  params.regex.motionregs        =  '^rp.*\.txt$';  % txt file with motion regressors (from realignment preproc step)
  params.regex.contrasts         = '^con.*\.nii$'; % contrast images

  %% Numbers
  % numbers
  params.num.subjects   = 32; % number of subjects (o rly)
  params.num.runs       =  6; % number of runs
  params.num.conditions =  8; % number of conditions
  params.num.motionregs =  6; % number of motion regressors

  params.num.badsSubjects = [28];
  params.num.goodSubjects = 1:params.num.subjects;
  params.num.goodSubjects(params.num.badsSubjects) = [];

  params.num.runIDs    =    [8,12,16,20,24,28];
  params.num.s22runIDs =    [10,14,18,22,26,30];
  %% Miscellaneous
  % EPI image options
  params.epi.voxelSize = [3.5 3.5 3.5];

  %% STEP0 : DICOM import
  params.import.dicom.root             = 'series'; % dunno
  params.import.dicom.outdir = [ params.dir.imDir 'TIMO001']; %NOTE obv. need to make this subject specific
  params.import.dicom.protfilter       =     '.*'; % protocol name filter
  params.import.dicom.convopts.format  =    'nii'; % output format (single nii vs nii + hdr)
  params.import.dicom.convopts.meta    =        0; % export metadata
  params.import.dicom.convopts.icedims =        0; % use ICEDims in filename (no). Useful if image sorting fails (due to multiple vols with identical file name)

  %% STEP1 : Realignment
  % estimation parameters
  params.reuw.estimate.quality   =     0.9; % quality
  params.reuw.estimate.sep       =       4; % separation (mm) between sampled points in ref img
  params.reuw.estimate.fwhm      =       5; % smoothing kernel prior to avg. fMRI: 5mm, PET: 7mm
  params.reuw.estimate.rtm       =       0; % num passes (fMRI: reg to first (0), PET: reg to mean (dunno val))
  params.reuw.estimate.einterp   =       2; % interpolation. 2= 2nd degree b-spline
  params.reuw.estimate.ewrap     = [0 0 0]; % wrapping
  params.reuw.estimate.weight    =      ''; % weighting. only necessary when extra brain motion (e.g. speech)
  % unwarp-estimate parameters
  params.reuw.unwarpest.basfcn   =   [12 12]; % basis functions
  params.reuw.unwarpest.regorder =         1; % regularisation default: 1st spatial derivative of estimated field
  params.reuw.unwarpest.lambda   =    100000; % regularisation factor
  params.reuw.unwarpest.jm       =         0; % jacobian intensity modulation for field estimation
  params.reuw.unwarpest.fot      =     [4 5]; % first order effects. default: pitch & roll. 1,2,3: translation x,y,z. 4,5,6: rotation x,y,z
  params.reuw.unwarpest.sot      =        []; % second order effects
  params.reuw.unwarpest.uwfwhm   =         4; % smoothing for unwarp
  params.reuw.unwarpest.rem      =         1; % re-estiamtion of movement parameters at each unwarping iteration
  params.reuw.unwarpest.noi      =         5; % number of unwarping iterations
  params.reuw.unwarpest.expround = 'Average'; % taylor-expansion point. avg gives best var reduction
  % unwarp-reslice parameters
  params.reuw.unwarpresl.uwwhich =   [2 1]; % images to reslice. 2,1: all images and mean image of resliced images (meanuf)
  params.reuw.unwarpresl.rinterp =       4; % interpolation. 4th deg b-spline
  params.reuw.unwarpresl.wrap    = [0 0 0]; % wrapping. no wrapping
  params.reuw.unwarpresl.mask    =       1; % masking due to motion
  params.reuw.unwarpresl.prefix  =     'u'; % .nii img file prefix

  %% STEP2 : Slice-Time Correction
  params.st.nslices  =        32; % number of slices
  params.st.tr       =         2; % TR (seconds)
  params.st.ta       =    1.9375; % TA = TR-(TR/nlices)
  params.st.so       = [32:-1:1]; % slice order. here: descending
  params.st.refslice =        16; % reference slice. here: middle slice
  params.st.prefix   =       'a'; % .nii img file prefix

  %% STEP3 : Coregistration
  params.coreg.estimate.cost_fun =   'nmi'; % objective function. normalised mutual information
  params.coreg.estimate.sep      =   [4 2]; % separation (mm): avg dist between sample points vector: coarse followed by fine registration
  params.coreg.estimate.tol      = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001]; % tolerances, i.e. accuracy for each parameter
  params.coreg.estimate.fwhm     =   [7 7]; % histogram smoothing. gaussian smooth applied to 256x256 joint hist.
  params.coreg.reslice.interp    =       4; % 4th deg b-spline interpolation
  params.coreg.reslice.wrap      = [0 0 0]; % no wrapping
  params.coreg.reslice.mask      =       0; % no masking
  params.coreg.reslice.prefix    =     'r'; % .nii img file prefix

  %% STEP4 : Normalisation
  params.norm.estimate.biasreg  =                            0.0001; % bias regularisation (very light)
  params.norm.estimate.biasfwhm =                                60; % gaussian smoothness of bias (mm)
  params.norm.estimate.tpm      = [params.dir.spmDir 'tpm/TPM.nii']; % tissue probability atlas
  params.norm.estimate.affreg   =                             'mni'; % affine regularisation. (ICBM space template - europ. brains)
  params.norm.estimate.reg      =            [0 1e-03 0.5 0.05 0.2]; % warping regularisation
  params.norm.estimate.fwhm     =                                 0; % smoothness (mm). fMRI: 0. PET/SPECT: 5mm
  params.norm.estimate.samp     =                                 3; % sampling distance between points for model estimation
  params.norm.write.bb          =          [-78 -112 -70; 78 76 85]; % bounding box of volume (relative to anterior commisure)
  params.norm.write.vox         =              params.epi.voxelSize; % voxel size (mm)
  params.norm.write.interp      =                                 4; % interpolation for image sampling and transformation in new space (4th deg. b-spline)
  params.norm.write.prefix      =                               'w'; % .nii img file prefix

  %% STEP5 : Smoothing
  params.smth.fwhm   = [8 8 8]; % gaussian smoothing kernel (mm)
  params.smth.dtype  =       0; % data type of the output images (use same as input)
  params.smth.im     =       0; % implicit mask (e.g. 0 for int and NaN for floats). 0: mask of input not preserved. 1: mask preserved
  params.smth.prefix =     's'; % .nii img file prefix

  %% STEPN : Tissue segmentation
  % data
  params.seg.channel.biasreg  = 0.001; % bias regularisation (very light)
  params.seg.channel.biasfwhm =    60; % gaussian smoothness of bias (mm)
  params.seg.channel.write    = [0 0]; % save bias correct and/or field (or nothing)
  % tissues
  params.seg.tissue(1).tpm{1} = [params.dir.spmDir 'tpm/TPM.nii,1']; % tissue prob map for specific tissue
  params.seg.tissue(1).ngaus  =     1; % number of gaussians (depends on tissue type)
  params.seg.tissue(1).native = [1 0]; % native tissue: create tissue class image (c*) in alignment with orig image
  params.seg.tissue(1).warped = [0 0]; % warped tissue. normalised version of tissue class image (currently not required)
  params.seg.tissue(2).tpm{1} = [params.dir.spmDir 'tpm/TPM.nii,2']; % ditto
  params.seg.tissue(2).ngaus  =     1;
  params.seg.tissue(2).native = [1 0];
  params.seg.tissue(2).warped = [0 0];
  params.seg.tissue(3).tpm{1} = [params.dir.spmDir 'tpm/TPM.nii,3']; % guess what..
  params.seg.tissue(3).ngaus  =     2;
  params.seg.tissue(3).native = [1 0];
  params.seg.tissue(3).warped = [0 0];
  params.seg.tissue(4).tpm{1} = [params.dir.spmDir 'tpm/TPM.nii,4'];
  params.seg.tissue(4).ngaus  =     3;
  params.seg.tissue(4).native = [1 0];
  params.seg.tissue(4).warped = [0 0];
  params.seg.tissue(5).tpm{1} = [params.dir.spmDir 'tpm/TPM.nii,5'];
  params.seg.tissue(5).ngaus  =     4;
  params.seg.tissue(5).native = [1 0];
  params.seg.tissue(5).warped = [0 0];
  params.seg.tissue(6).tpm{1} = [params.dir.spmDir 'tpm/TPM.nii,6'];
  params.seg.tissue(6).ngaus  =     2;
  params.seg.tissue(6).native = [0 0];
  params.seg.tissue(6).warped = [0 0];

  % warping and MRF
  params.seg.warp.mrf         =     1; % markov random field cleanup strength (applied to tissue class images)
  params.seg.warp.cleanup     =     1; % extract brain from segmented images. routine strength (1=light)
  params.seg.warp.reg         = [0 0.001 0.5 0.05 0.2]; % warping regularisation
  params.seg.warp.affreg      = 'mni'; % affine regularisation. (ICBM space template - europ. brains)
  params.seg.warp.fwhm        =     0; % smoothness (mm). fMRI: 0. PET/SPECT: 5mm
  params.seg.warp.samp        =     3; % sampling distance between points for model estimation
  params.seg.warp.write       = [0 0]; % deformation fields (none)
