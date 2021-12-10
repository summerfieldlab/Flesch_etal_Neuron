function [idxMat,volMat] = fmri_mask_genGroupMask(doExport)
  %% fmri_mask_genGroupMask()
  %
  % generates group-level mask based on overlapping Ss masks
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  if ~exist('doExport','var')
    doExport = 1;
  end
  params = rsa_compute_setParams();

  fileDir = params.dir.inDir;
  maskDir = params.dir.subDir.SPM;
  outDir  = params.dir.maskDir;

  maskIDs = [];
  for ii = 1:length(params.num.goodSubjects)
    subID = params.num.goodSubjects(ii);
    subName = set_fileName(subID);
    subPath = [fileDir subName '/' maskDir];
    indSet = fmri_mask_mask2ind('mask.nii',subPath);
    maskIDs(subID,:) = indSet.all;
  end

  % extract indices for voxels that contain data of all participants
  groupMask = find(~any(isnan(maskIDs),1)>0);

  % put em in a matrix:
  [x,y,z] = ind2sub(indSet.dim,groupMask');
  idxMat = [x y z]';

  % generate volume (3D matrix of voxel intensities)
  volMat = fmri_volume_genVolume(indSet.dim,idxMat,ones(size(idxMat,2),1));

  if doExport
    % export volume to nifti file
    fName = [outDir params.names.groupMask '.nii'];
    descript = 'group-level mask';
    dType = 2; % uint8
    fmri_io_mat2nifti(volMat,fName,descript,dType);
  end

end
