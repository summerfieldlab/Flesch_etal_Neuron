function rsa_disp_showCorrs_ROI_SVD(nDims,maskName,labels,maskLabel)
  %% rsa_disp_showCorrs_ROI_SVD
  %
  % plots model correlations for different dimensionalities of the
  % input data
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  params = rsa_roi_params();
  sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));

  mu = [];
  er  = [];
  pvals = [];
  tvals = [];
  xLabels = {' '};
  %% import data
  for d = 1:nDims
    % load betas
    load(['groupAvg_' params.names.betasOut '_regress' '_' num2str(d) 'D_' maskName '.mat']);
    mu(d,:) = mean(results.corrs,1);
    err(d,:) = sem(results.corrs,1);
    % load sigtest results
    load(['groupAvg_STATS_' params.names.betasOut '_' num2str(d) 'D_' maskName '.mat']);
    pvals(d,:) = stats.p;
    tvals(d,:) = stats.t;
    xLabels{d+1} = [num2str(d)];
  end
%   xLabels{end+1} = 'full dataset';
%   xLabels{end+1} = ' ';
%   
%   load(['groupAvg_modelBetas_cval_slRSA_paper__' maskName '.mat']);
%   mu(end+1,:) = mean(results.corrs,1);
%   err(end+1,:) = sem(results.corrs,1);
%   
%   % load sigtest results
%   load(['groupAvg_STATS_modelBetas_cval_slRSA_paper__' maskName '.mat']);
%   pvals(end+1,:) = stats.p;
%   tvals(end+1,:) = stats.t;

  %% plot betas
  figure();set(gcf,'Color','w');
  % subplot(2,1,1);
  b = errorbar(mu,err,'LineWidth',1);
  for ii = 1:length(b)
    b(ii).MarkerFaceColor = 'auto';
    b(ii).MarkerSize = 4;
    b(ii).Marker = 'o';
    b(ii).CapSize = 0;
  end
  box off;
  xlim([0,size(mu,1)+1])
  ylim([-0.03,0.07])
  % set(gca,'XTick',0:size(mu,1));
  % set(gca,'XTickLabel',xLabels);
  hold on; 
  plot(get(gca,'XLim'),[0,0],'k--');
  legend(b,labels,'Location','NorthWest','Box','off');
  xlabel('\rm retained components');
  ylabel('\rm \beta estimate (a.u.)');
  title(['Truncated SVD, ' maskLabel]);

  % %% plot t scores
  % % figure();set(gcf,'Color','w');
  % subplot(2,1,2);
  % % yyaxis right
  % b = bar(tvals,'LineWidth',1.5,'EdgeColor','none');

  % box off;
  % xlim([0,size(mu,1)+1])
  % ylim([-2,10]);
  % set(gca,'XTick',0:size(mu,1));
  % set(gca,'XTickLabel',xLabels);
  % hold on;plot(get(gca,'XLim'),[4,4],'k--');
  % legend(b,labels,'Location','NorthWest');
  % set(gcf,'Position',[675,567,1035,1016]);
  % % set(gcf,'Position',[675,567,1035,408]);
  % xlabel('\bf data dimensionality');
  % ylabel('\bf t-Score');


end
