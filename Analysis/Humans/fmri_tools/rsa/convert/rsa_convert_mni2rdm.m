function groupRDM = rsa_convert_mni2rdm(coords,sphereSize)
  %% rsa_convert_mni2rdm()
  %
  % given a set of MNI coordinates,
  % this function extracts corresponding single-subject RDMs
  % and returns an averaged group RDM for said coordinates
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

  params = rsa_rdms_setParams();

  if ~exist('sphereSize','var')
    sphereSize = 2;
  end

  % load mask
  maskImg = spm_vol(['groupMask_rsaSearchlight.nii']);

  % get linear indices
  indices = fmri_mask_mni2roi(coords,sphereSize,maskImg);

  groupRDM = [];
  % % average rdms
  % for subID = 1:params.num.subjects
  %   disp(['processing subject ' num2str(subID)]);
  %   load(['rdmSet_searchlight_sub' num2str(subID)]);
  %   groupRDM(subID,:,:) = squeeze(mean(rdmSet(indices,:,:),1));
  % end

  load(['rdmSet_searchlight_avg']);
  % return avg group ROI RDM
  groupRDM = squeeze(nanmean(rdmSet(indices,:,:),1));
end
