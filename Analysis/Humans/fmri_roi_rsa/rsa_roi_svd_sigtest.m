function stats = rsa_roi_svd_sigtest(maskName,nDims)
  %% rsa_roi_svd_sigtest()
  %
  % performs statistical inference on
  % correlation coefficients within ROIs
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

  params = rsa_roi_params();
  % load correlations
  load(fullfile(params.dir.outDir, ['groupAvg_' params.names.betasOut '_regress' '_' num2str(nDims) 'D_'   maskName '.mat']));

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
  save(fullfile(params.dir.outDir, ['groupAvg_STATS_' params.names.betasOut '_' num2str(nDims) 'D_' maskName '.mat']),'stats');

end
