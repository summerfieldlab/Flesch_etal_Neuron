function [linIdces,volMat] = fmri_mask_genSphericalMask(centroid,radius,globalMask)
  %% rsa_mask_genSphericalMask()
  %
  % creates mat of indices for spherical searchlight
  % requires indices of whole-brain volume as input
  %
  % INPUTS:
  % - centroid: [x,y,z] location of mask's centre
  % - radius:           radius of spherical mask (in voxels)
  % - globalMask:       volume of whole-brain mask
  %
  % OUTPUTS:
  % - linIdces: linear indices of mask
  % - volMat: volume for masking (1=sphere, nan=elswhere)
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford



  % obtain coordinates for grp mask
  [x,y,z] = ind2sub(size(globalMask),[1:prod(size(globalMask))]');
  brainXYZ = [x,y,z];
  % obtain linear indices of spherical mask
  linIdces = find(pdist2(brainXYZ,centroid)<=radius);
  [x,y,z] = ind2sub(size(globalMask),linIdces);
  matIdces = [x,y,z]';
  % generate volume (3D matrix of voxel intensities with ones inside sphere and 0s everywhere else)
  volMat = fmri_volume_genVolume(size(globalMask),matIdces,ones(length(linIdces),1));
  % discard extra-brain voxels if sphere not entirely within brain volume
  volMat = volMat .* globalMask;
  linIdces = find(volMat(:)==1);
end
