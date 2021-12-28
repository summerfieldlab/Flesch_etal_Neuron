function mante_disp_perm_rdms(results)
%% mante_disp_permrdms(results)
%
% visualises average of simulated null rdms 
% 
% Timo Flesch,  2020


%% raw RDMs and mds projections
figure(); set(gcf,'Color','w');
for ii = 1:2
    subplot(1,2,ii);
    aux_showRDM(squeeze(nanmean(results(ii).rdms,1)));
    title(['\rm Randomised Data - ' results(ii).name]);
    box off;
    axis square;            
    cb = colorbar();
    ylabel(cb,'dissimilarity');
    
end
set(gcf,'Position',[141   556   754   311]);
figure(); set(gcf,'Color','w');
for ii = 1:2
    subplot(1,2,ii);
    aux_showMDS(squeeze(nanmean(results(ii).rdms,1)),3);
    title(['\rm Randomised Data - ' results(ii).name]);
    box off;
    axis square;                
end
set(gcf,'Position',[141   556   754   311]);
