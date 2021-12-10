function rsa_disp_singular_values()
  %% rsa_disp_singular_values
  %
  % plots group average singular values in terms of
  % in terms of their explained (cumulative) variance
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford


  masks = {'leftEVC_mod1_','rightEVC_mod1_', 'rightPAR_mod3_','rightIFG_mod3_'};
  load(['groupAvg_singular_values_' masks{1} '.mat' ]);
  results_levc = results;
  load(['groupAvg_singular_values_' masks{2} '.mat' ]);
  results_revc = results;
  load(['groupAvg_singular_values_' masks{3} '.mat' ]);
  results_rangu = results;
  load(['groupAvg_singular_values_' masks{4} '.mat' ]);
  results_rifg = results;

  %% calculate explained variance
  csve_levc = [];
  csve_revc = [];
  csve_rangu = [];
  csve_rifg = [];
  for ii = 1:size(results_levc.singular_values,1)
    sv = results_levc.singular_values(ii,:);
    ve = (sv.^2)./sum(sv.^2);
    csve_levc(ii,:) = cumsum(ve);
    
    sv = results_revc.singular_values(ii,:);
    ve = (sv.^2)./sum(sv.^2);
    csve_revc(ii,:) = cumsum(ve);

    sv = results_rangu.singular_values(ii,:);
    ve = (sv.^2)./sum(sv.^2);
    csve_rangu(ii,:) = cumsum(ve);
    
    sv = results_rifg.singular_values(ii,:);
    ve = (sv.^2)./sum(sv.^2);
    csve_rifg(ii,:) = cumsum(ve);
  end
  


  figure();set(gcf,'Color','w');
  b = {};
  b{1} = helper_plot_scree(csve_levc,[0 .4 0]);
  b{1}.CapSize = 0;
  hold on;
  b{2} = helper_plot_scree(csve_revc,[0 .6 0]);
  b{2}.CapSize = 0;
  hold on;
  b{3} = helper_plot_scree(csve_rangu,[1 .6 0]);
  b{3}.CapSize = 0;
  hold on;
  b{4} = helper_plot_scree(csve_rifg, [.5 0 0]);
  b{4}.CapSize = 0;

  grid off
  set(gcf,'Color','w');
  box off
  ylim([0,1])
  % xlim([0,30])
  set(gca,'YTick',0:.2:1);
  set(gca,'YTickLabel',get(gca,'YTick').*100);  
  lgd  = legend({'leftEVC', 'rightEVC','rightAngularGyrus','rightDLPFC'},'Location','SouthEast');
  set(lgd,'Box','off')
  xlabel('\rm number of components');
  ylabel('\rm cumulative explained variance (%)');
  set(gcf,'Position',[ 1091,   373,    337,    286 ]);

end



function b = helper_plot_scree(data, col)
    % if ~exist('col','var')
    %     col = 'auto';
    % end

    sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));

    e = sem(data,1);
    m = mean(data,1);
    
    b = errorbar(1:length(m),m,e,'LineWidth',1,'Color',col)
    for ii = 1:length(b)
        b(ii).MarkerFaceColor = col;
        b(ii).MarkerSize = 2;
        b(ii).Marker = 'diamond';
    end
end
