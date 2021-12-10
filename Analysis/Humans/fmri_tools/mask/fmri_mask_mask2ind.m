function indices = fmri_mask_mask2ind(maskName,maskDir)
  %% fmri_mask_mask2ind(maskName,maskDir)
  %
  % computes indices from a mask .nii file
  %
  % INPUTS;
  % - maskName: filename of mask
  % - maskDir:  path to folder containing mask
  %
  % OUTPUTS;
  % - indices: struct with fields mat and dim
  % - .mask:   3D matrix with indices of non-NaN voxels
  % - .dim:    Dimensionality of image (useful for initialisation)
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab,
  % University of Oxford
  
  if exist('maskDir')
    % full file
    mask = [maskDir,maskName];
  else
    mask = maskName;
  end
  indices = struct();
  % import mask as voxel array
  M = spm_read_vols(spm_vol(mask),1);
  % get indices of brain voxels
  idces = find(M>0);
  % divide indices in x,y,z coordinates
  [x,y,z] = ind2sub(size(M),idces);
  % put em in a matrix:
  indMat = [x y z]';
  indices.mat = indMat;
  indices.all = M(:);
  indices.dim = size(M);

end
