function mante_disp_regressionresults_svd(params,betas,betas_perm)
    %% mante_disp_regressionresults_svd(params,betas,betas_perm)
    % 
    % display results of svd rsa
    % 
    % Timo Flesch, 2021

    modelrdms = mante_gen_modelrdms(params);
    modelnames = {modelrdms.name};
    % cols = {'magenta','green','blue','cyan'}
    figure(1); set(gcf,'Color','w');
    for i = 1:1
        % subplot(1,2,i); 
        for j = 1:size(betas,2)
            
            fh(j) = plot(1:size(betas,3),squeeze(betas(i,j,:))','LineWidth',2);
            hold on
            
            aux_ts_shadederrorbar(1:size(betas_perm,4),squeeze(betas_perm(i,:,j,:)),fh(j).Color,1,'2std',0.2,'--');
        end
        plot(get(gca,'XLim'),[0,0],'k--')
        
        
        xlabel('number of retained components')
        ylabel('\beta estimate (a.u.)')
        set(gcf,'Color','w')
        box off        
        title(['\rm Model RSA, Monkey ' params.analysis.monknames{i}]);
        
        legend(fh, modelnames,'Box','off', 'Location', 'NorthEastOutside');
        
    
    end
    % set(gcf,'Position', [248, 666, 1544, 334]);

end