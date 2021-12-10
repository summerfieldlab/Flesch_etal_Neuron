function new = fmri_io_reslice(vol,ref,doSave)
  %% fmri_io_reslice()
  %
  % interpolates vol such that its dimensions match with ref
  % Inputs:
  % vol: the volume to reslice (spm_vol format)
  % ref: the reference volume (spm_vol format)
  %
  % Outputs:
  % new: the resliced and masked volume
  %
  % Note:
  % shortened version of Jan's scan_nifti_reslice.m
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  if ~exist('doSave','var')
    doSave = 0;
  end

  d = [4*[1 1 1]' [0;0;0]];
  C = spm_bsplinc(vol, d);

  % jdobalaguer's reslicing magic:
  new = zeros(ref.dim);
  [x1,x2] = ndgrid(1:ref.dim(1),1:ref.dim(2));
  for x3 = 1:ref.dim(3)
      M = inv(ref.mat\vol.mat);
      y1   = M(1,1)*x1+M(1,2)*x2+(M(1,3)*x3+M(1,4));
      y2   = M(2,1)*x1+M(2,2)*x2+(M(2,3)*x3+M(2,4));
      y3   = M(3,1)*x1+M(3,2)*x2+(M(3,3)*x3+M(3,4));
      new(:,:,x3)  = spm_bsplins(C, y1,y2,y3, d);
  end

  % convert to integer array
  new = cast(new,'uint8');
  % include only brain voxels
  refMask = cast(spm_read_vols(ref),'uint8');
  new = new .* refMask;

  if doSave
    fmri_io_mat2nifti(new,['r_' vol.fname],vol.descrip,2); % export as unsigned integer array
  end


end
