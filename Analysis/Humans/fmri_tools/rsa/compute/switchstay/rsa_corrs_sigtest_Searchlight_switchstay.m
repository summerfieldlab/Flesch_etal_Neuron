function results = rsa_corrs_sigtest_Searchlight_switchstay(phaseID)
  %% rsa_corrs_sigtest_Searchlight()
  %
  % performs statistical inference on
  % maps of correlation coefficients
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

  params = rsa_compute_setParams_switchstay(phaseID);
  corrs = load(['modelbetas_searchlight_' phaseID '_' params.names.modelset  '_allsubs_masked']);
  corrs = corrs.betas;
  dimsAll = size(corrs);
  dims = dimsAll(3:end);

  results = struct();
  results.p = nan(dimsAll(2:end));
  switch params.statinf.method
  case 'signrank'
    results.z = nan(dimsAll(2:end));
  case 'ttest'
    results.t = nan(dimsAll(2:end));
  end
  for modID = 1:size(corrs,2)
    corrs_redux = squeeze(corrs(:,modID,:));
    % obtain indices of voxels that contain values,
    % as I don't want to apply sigtests to nans (saves time, obs)
    corrs_idces = find(~any(isnan(corrs_redux),1));
    corrs_redux = corrs_redux(:,corrs_idces);
    % convert linear to 3d indices
    [x,y,z] = ind2sub(dims,corrs_idces);
    % perform the tests
    disp(['processing model ' num2str(modID)]);
    for voxID = 1:length(corrs_idces)
      switch params.statinf.method
      case 'signrank'
        [p,~,s] = signrank(corrs_redux(:,voxID),0,'method','approximate','tail',params.statinf.tail);
        results.p(modID,x(voxID),y(voxID),z(voxID)) = p;
        results.z(modID,x(voxID),y(voxID),z(voxID)) = s.zval;
      case 'ttest'
        if params.statinf.doFisher
          corrs_redux(:,voxID) = atanh(corrs_redux(:,voxID));
        end
        [~,p,~,s] = ttest(corrs_redux(:,voxID),0,'tail',params.statinf.tail);
        results.p(modID,x(voxID),y(voxID),z(voxID)) = p;
        results.t(modID,x(voxID),y(voxID),z(voxID)) = s.tstat;
      end
    end
  end
  results.params = params.statinf;
  save(['modelbetas_searchlight_' phaseID '_' params.names.modelset '_allsubs_masked_STATS.mat'],'results');

end
