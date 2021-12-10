function [rdmSet,avgRDM] = rsa_compute_averageCvalRDMs(rdm,nRuns,nConds)
%% rsa_compute_averageCvalRDMs(rdm,nRuns,nConds)
%
% averages dissimilarities of lower triangular form
% of between-run-rdm
%
% Timo Flesch, 2019
% Human Information Processing Lab
% University of Oxford
  idx = 1;
  for ii = 1:nRuns-1
    yStart = ii*nConds+1;
    xStart = 1 + (ii-1)*nConds;
    for jj = ii+1:nRuns
      rdmSet(idx,:,:) = rdm(yStart:(yStart+nConds-1),xStart:(xStart+nConds-1));
      yStart = yStart+nConds;
      idx = idx+1;
    end
  end
  avgRDM = squeeze(nanmean(rdmSet,1));
end
