function mante_disp_rdms(results,params)
%% mante_disp_rdms(results)
%
% displays raw rdms and MDS projections 
% for both monkeys
% 
% Timo Flesch,  2020

cd(params.dir.figures);
%% raw RDMs and mds projections
figure(); set(gcf,'Color','w');
for ii = 1:2
    subplot(1,2,ii);
    aux_showRDM(results(ii).rdm);
    title(['\rm' results(ii).name]);
    box off;
    axis square;            
    cb = colorbar();
    ylabel(cb,'dissimilarity');    
end
set(gcf,'Position',[141   556   754   311]);
% savefig('rdms_monkeys.fig')
figure(); set(gcf,'Color','w');
for ii = 1:2
    subplot(1,2,ii);
    aux_showMDS(results(ii).rdm,2);
    title(['\rm' results(ii).name]);
    box off;
    axis square;                
end
set(gcf,'Position',[141   556   754   311]);
% savefig('mds_monkeys.fig')
cd(params.dir.project);
