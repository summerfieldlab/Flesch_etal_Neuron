function fmri_io_mat2nifti(volMat,fName,descript,dType,V)
  %% fmri_io_mat2nifti()
  %
  % exports matrix (T-map, for instance)
  % as .nii file for visualisation with favourite image viewer
  % (bspmview, xjview and the like)
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford
  
  if ~exist('V','var')
    params = fmri_io_setParams();
    % create volume structure
    V = params.vol;
  end
  V.private = [];
  V.fname  = fName;
  V.descrip = descript;
  V.dt(1) = dType;
  spm_write_vol(V,volMat);

end
