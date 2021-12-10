function stats = rsa_corrs_sigtest_ROI_switchstay(maskName,phaseName)
  %% rsa_corrs_sigtest()
  %
  % performs statistical inference on
  % correlation coefficients within ROIs
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

  params = rsa_corrs_setParams_switchstay(phaseName);
  % load correlations
  load(['groupAvg_' params.names.corrsOut '_' maskName '.mat']);

  for modID = 1:size(results.corrs,2)
      switch params.statinf.method
      case 'signrank'
        [p,~,s] = signrank(results.corrs(:,modID),0,'method','approximate','tail',params.statinf.tail);
        stats.p(modID) = p;
        stats.z(modID) = s.zval;
      case 'ttest'
        if params.statinf.doFisher
          results.corrs(:,modID) = atanh(results.corrs(:,modID));
        end
        [~,p,~,s] = ttest(results.corrs(:,modID),0,'tail',params.statinf.tail);
        stats.p(modID) = p;
        stats.t(modID) = s.tstat;
      end
  end
  stats.params = params.statinf;
  save(['groupAvg_STATS_' params.names.corrsOut '_' maskName '.mat'],'stats');

end
