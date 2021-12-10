function rsa_convert_struct2mat()
  %% rsa_convert_struct2mat()
  %
  % imports rdms computed with tdt toolbox (hebart et al) and
  % stores them in a subject specific voxel-x-dim1-x-dim2 matrix.
  % adds entry with linear indices
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  params = rsa_rdms_setParams();


  for subID = 1:params.num.subjects
    subStr = sprintf('TIMO%03d',subID);
    results = load([params.dir.inDir subStr '/' params.dir.subDir.RDM 'res_other_average.mat']);
    results = results.results;
    brainRDMs = results.other_average.output;
    rdmSet = nan([prod(results.datainfo.dim),size(brainRDMs{1})]);
    for ii = 1:length(results.mask_index)
      rdmSet(results.mask_index(ii),:,:) = brainRDMs{ii};
    end
    save(['rdmSet_searchlight_sub' num2str(subID)],'rdmSet','-v7.3');
  end

end
