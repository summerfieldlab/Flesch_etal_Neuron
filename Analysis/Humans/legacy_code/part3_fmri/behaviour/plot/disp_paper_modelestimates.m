function disp_modelestimates(results_day1,results_scan)
%%  disp_results(results_day1,results_scan)
%
% plots estimated parameters
%
% Timo Flesch, 2018,
% Human Information Processing Lab,
% Experimental Psychology Department
% University of Oxford
badsubs_scan = [19,28]; % no training data available (performed at 81 and 64 % in scanner)
% badsubs_train = [27];


sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));


phase_name      = {'test_{baseline}', 'test_{scan}'};
thetas         = {'slope','offset','lapse','bias'};



results_all{1} = results_day1.p2;
results_all{2} = results_scan;
for ii = 1:length(thetas)
    % tmp = results_all{1}.(thetas{ii});
    % tmp([badsubs_train])  = [];
    % results_all{1}.(thetas{ii}) = tmp;

    tmp = results_all{2}.(thetas{ii});
    tmp([badsubs_scan])  = [];
    results_all{2}.(thetas{ii}) = tmp;

end

colvals = linspace(0.4,0.6,length(results_all));

for ii = 1:length(thetas)
    figure();set(gcf,'Color','w');
    for jj = 1:length(results_all)
        b = bar(jj,mean(results_all{jj}.(thetas{ii})));
        b.FaceColor = [1,1,1].*colvals(jj);
        hold on;
        plotSpread_hack(results_all{jj}.(thetas{ii}),'distributionMarkers','o','xValues',jj);
        eb = errorbar(jj,mean(results_all{jj}.(thetas{ii})),sem(results_all{jj}.(thetas{ii}),1),'LineWidth',2.5);
        eb.Color = [0,0,0];
    end
    % keyboard
    plot([1;2],[results_all{1}.(thetas{ii}),results_all{2}.(thetas{ii})]','Color',[.8 .8 .8],'LineStyle','-');
    [pval,~] = signrank(results_all{1}.(thetas{ii}),results_all{2}.(thetas{ii}));
    sigstar([1,2],pval);
    box off;
    set(gca,'XTick',1:2);
    set(gca,'XTickLabel',phase_name);
    xlabel('Phase');
    ylabel(thetas{ii});
    set(gcf,'Position', [2637, 358, 281, 283]);
end
%
% plot([1;2],[results_all{jj};results_all{2}],'Color',[.8 .8 .8],'LineStyle','-');
% [~,pval] = ttest(results_all{jj},results_all{2});
% sigstar([1,2],pval);
% box off;
% set(gca,'XTick',1:length(results_all));
% set(gca,'XTickLabel',phase_name);
% set(gca,'TickDir','out');
% ylabel('\bf Accuracy [%]');
% set(gca,'YTick',[0:.2:1])
% set(gca,'YTickLabel',[0:20:100])
% ylim([0.3,1.2]);
% plot(get(gca,'XLim'),[.5 .5],'k--');
% xlim([0,length(results_all)+1]);
end
