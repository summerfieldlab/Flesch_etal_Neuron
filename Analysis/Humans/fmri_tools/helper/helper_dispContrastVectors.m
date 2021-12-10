function helper_dispContrastVectors(contrasts,params)
  %% HELPER_DISPCONTRASTVECTORS(CONTRASTS,PARAMS)
  %
  % visualises all generated contrast vectors
  % Timo Flesch, 2018

  figure();set(gcf,'Color','w');
  % imagesc(contrasts.T.vectors);
  % colormap('gray');
  % set(gca,'YTickLabel',contrasts.T.labels);
  % xlabel('\bf Column');
  % ylabel('\bf Contrast Name');
  % cb = colorbar();
  % ylabel(cb,'Value');
  % title({'\bf Contrast Vectors'; ['\rm' 'conditions: ' num2str(params.num.conditions) ', motionregs: ' num2str(params.num.motionregs) ', runs: ' num2str(params.num.runs)]});
  % set(gca,'CLim',[-1,1]);


  for ii = 1:size(contrasts.T.vectors,1)
    subplot(size(contrasts.T.vectors,1),1,ii);
    b = bar(contrasts.T.vectors(ii,:));
    b.FaceColor= [1,1,1].*0.5;
    limval = max(abs(contrasts.T.vectors(ii,:)));
    set(gca,'YLim',[-1,1].*(limval*1.2));
    ylabel(['\bf' contrasts.T.labels{ii}]);
    xlabel('\bf Column');
    title({'\bf Contrast Vectors'; ['\rm' 'conditions: ' num2str(params.num.conditions) ', motionregs: ' num2str(params.num.motionregs) ', runs: ' num2str(params.num.runs)]});
    box off;
    grid on;
  end
end
