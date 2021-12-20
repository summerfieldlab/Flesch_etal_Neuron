function disp_modelcorrs(taus_day1,taus_scan)
%% disp_modelcorrs(taus_day1,taus_refresher,taus_scan)
%
% displays mean taus for each phase
%
% Timo Flesch, 2018,
% Human Information Processing Lab,
% Experimental Psychology Department
% University of Oxford

badsubs = [19,28]; % no training data available (performed at 81 and 64 % in scanner)

sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));

% phaseName    = {'training_{baseline}', 'test_{baseline}', 'training_{main,task1}', 'training_{main,task2}', 'training_{refresher}', 'test_{scan}'};
phaseName      = {'test_{baseline}', 'test_{scan}'};


taus_all = {};
taus_all{1} = taus_day1.p2.blocked.both;
taus_all{2} = taus_scan.blocked.both;
taus_all{2}([badsubs],:)  = [];

colvals = linspace(0.4,0.6,2);

figure();set(gcf,'Color','w');
e = [sem(taus_all{1},1);sem(taus_all{2},1)];
m = [mean(taus_all{1},1);mean(taus_all{2},1)];
b = barwitherr(e,m);
b(1).FaceColor = [1,1,1].*colvals(1);
b(2).FaceColor = [1,1,1].*colvals(2);
hold on;
% factorised model
plotSpread_hack([taus_all{1}(:,1),taus_all{2}(:,1)],'distributionMarkers','o','xValues',[1,2],'xOffset',-0.15)
% linear model
plotSpread_hack([taus_all{1}(:,2),taus_all{2}(:,2)],'distributionMarkers','o','xValues',[1,2],'xOffset',+0.15)

box off;
set(gca,'XTick',1:length(taus_all));
set(gca,'XTickLabel',phaseName);
set(gca,'TickDir','out');
ylabel('\bf Correlation [\tau_{a}]');
ylim([-0.2,0.8]);
xlim([0,length(taus_all)+1]);
legend([b(1),b(2)],{'Factorised Model','Linear Model'},'Location','NorthEastOutside')
legend('boxoff');

end
