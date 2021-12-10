function scores = rsa_helper_rdmReplicability()
  %% rsa_helper_rdmReplicability()
  %
  % computes leave-one-subejct-out RDM correlations
  % as estimate of between-subject replicability.
  % Useful to compare different methods with each other
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  params = rsa_corrs_setParams();

  distMeasures = {'EuclDist','CorrDist','MahalDist_roi','WhitenCorrDist','Crossnobis_roi'};
  labels = {'euclidean','correlation','whitened euclidean','whitened correlation','crossnobis'};
  rois = {'BA17','BA18','BA19','IT','ACC'};

  rdmSet = [];
  grpDir = [params.dir.inDir params.dir.subDir.GRP];

  for rID = 1:length(rois)
    for dID = 1:length(distMeasures)
      % collect all RDMs in set
      for subID = 1:params.num.subjects
        % navigate to subdir
        subStr = params.names.subjectDir(subID);
        subDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
        cd(subDir);
        load(['rdmSet_' distMeasures{dID} '_r_mask_wfu_' rois{rID}]);
        rdmSet(:,:,subID) = subRDM.rdm;
      end
      cd(grpDir);
      % normalise RDMs appropriately
      rdmVect = vectorizeRDMs(rdmSet); % 1-x-dissims-x-subs
      [~, nDissims, nSubjects]=size(rdmVect);
      switch params.corrs.method
      case 'pearson'
          rdmVect=rdmVect-repmat(nanmean(rdmVect,2),[1 nDissims 1]);
          rdmVect=rdmVect./repmat(std(rdmVect,[],2),[1 nDissims 1]);
          meanRDM=nanmean(rdmVect,3);
      case {'spearman','kendall'}
          rdmVect=reshape(tiedrank(reshape(rdmVect,[nDissims nSubjects])),[1 nDissims nSubjects]);
          meanRDM=nanmean(rdmVect,3);
      end
      % compute leave one subject out rdm correlation
      for subID = 1:params.num.subjects
        losoVect = rdmVect(1,:,:);
        losoVect(:,:,subID) = [];
        losoVect = nanmean(losoVect,3);
        rdmCorrs(rID,dID,subID) = compareDissims(losoVect,rdmVect(1,:,subID),params.corrs.method);
      end
    end
  end
  scores         =     struct();
  scores.corrs   =     rdmCorrs;
  scores.metrics = labels;
  scores.rois    =         rois;
  save('rdmReplicability_visual.mat','scores');

end



function r = compareDissims(dv1,dv2,method)

  switch method
    case 'kendall'
      v1 = dv1(:);
      v2 = dv2(:);
      r = rankCorr_Kendall_taua(v1,v2);
    case 'spearman'
      v1 = dv1(:);
      v2 = dv2(:);
      r = corr(v1,v2,'type','Spearman');
    case 'regression'
      v1 = zscore(dv1(:));
      v2 = zscore(dv2(:));
      r = regress(v1',v2');
    case 'pearson'
      v1 = zscore(dv1(:));
      v2 = zscore(dv2(:));
      r = corr(v1,v2,'type','Pearson');
  end
end
