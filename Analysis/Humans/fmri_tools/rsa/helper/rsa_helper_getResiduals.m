function r = rsa_helper_getResiduals(SPM,maskIndices,checkResults)
  %% r = rsa_helper_getResiduals(SPM)
  %
  % computes residuals from nifti files of single subject
  % returns them as matlab struct with linear indices
  % optional: checks whether they match the ones stored in ResMS.nii
  %
  % based on following script:
  % https://github.com/sjgershm/ccnl-fmri/blob/master/ccnl_get_residuals.m
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford


  if ~exist('checkResults','var')
    checkResults = 0;
  end

  % import (masked) voxel responses
  if exist('maskIndices','var')
      keyboard
    Y = spm_data_read(SPM.xY.VY, maskIndices);
  else
    Y = spm_data_read(SPM.xY.VY);
    yShape = size(Y);
    Y = reshape(Y,[prod(yShape(1:3)),yShape(4)]);
    
    Y = Y';
  end


  % weight data and remove filter confounds (?)
  KWY = spm_filter(SPM.xX.K,SPM.xX.W*Y);

  % compute residuals
  r = spm_sp('r',SPM.xX.xKXs,KWY);


  if checkResults
    % compute sum of squared residuals
    ResSS = sum(r.^2);
    % scale by trace(RV)
    ResMSalt = ResSS/SPM.xX.trRV;
    % load mean squared residual image
    V = spm_vol(fullfile(SPM.swd,SPM.VResMS.fname));
    % only care about brain voxels
    ResMS = spm_data_read(V, mask);
    % test if equal
    assert(immse(ResMSalt, ResMS) < 1e-9, 'resids don''t match');
  end

end
