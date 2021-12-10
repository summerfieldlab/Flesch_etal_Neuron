function mante_disp_svd(params,results)
    %%mante_disp_svd(params,results)
    % 
    % display cumulative explained variance 
    % of responses in monkey PFC 
    % 
    % Timo Flesch, 2021

    figure(); set(gcf,'Color','w');
    h1 = plot(results(1).cumvar);    
    hold on
    h2 = plot(results(2).cumvar);
    legend([h1,h2],{['monkey ' params.analysis.monknames{1}],['monkey ' params.analysis.monknames{2}]} ,'Location','SouthEast','Box','off');
    % ylim([0,1]);
    xlim([1,72]);
    xlabel('component');
    ylabel('cumulative variance explained (%)');
    set(gca,'YTickLabel',get(gca,'YTick').*100);
    box off;
    title('\rm SVD on Response Matrix');
end
