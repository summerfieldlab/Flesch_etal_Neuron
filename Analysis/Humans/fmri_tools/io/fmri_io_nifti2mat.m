function vxlMat = fmri_io_nifti2mat(niiFile,niiDir,setNaN)
  %% vxlMat = fmri_io_nifti2mat()
  %
  % imports a .nii file (such as a T-Map)
  % and returns it as matrix
  % default: sets zeros to NaN
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford
  if ~exist('setNaN','var')
    setNaN = 1;
  end

  filePath = fullfile(niiDir,niiFile);
  vxlMat = spm_read_vols(spm_vol(filePath),setNaN);

end
