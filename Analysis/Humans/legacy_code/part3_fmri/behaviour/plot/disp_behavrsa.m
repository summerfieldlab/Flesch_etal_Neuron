function disp_behavrsa(betas_day1,betas_scan)
%% disp_modelcorrs(betas_day1,betas_refresher,betas_scan)
%
% displays mean betas for each phase
%
% Timo Flesch, 2018,
% Human Information Processing Lab,
% Experimental Psychology Department
% University of Oxford

badsubs = [19,28]; % no training data available (performed at 81 and 64 % in scanner)

sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));

% phaseName    = {'training_{baseline}', 'test_{baseline}', 'training_{main,task1}', 'training_{main,task2}', 'training_{refresher}', 'test_{scan}'};
phaseName      = {'test_{baseline}', 'test_{scan}'};


betas_all = {};
betas_all{1} = betas_day1.p2.blocked.both;
betas_all{2} = betas_scan.blocked.both;
betas_all{2}([badsubs],:)  = [];

colvals = linspace(0.4,0.6,2);

figure();set(gcf,'Color','w');
e = [sem(betas_all{1},1);sem(betas_all{2},1)];
m = [mean(betas_all{1},1);mean(betas_all{2},1)];
b = barwitherr(e,m);
b(1).FaceColor = [1,1,1].*colvals(1);
b(2).FaceColor = [1,1,1].*colvals(2);
b(1).EdgeColor = 'None';
b(2).EdgeColor = 'None';
hold on;
% factorised model
scatter(ones(length(betas_all{1}(:,1)),1).*(1-0.15),betas_all{1}(:,1)','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None');
scatter(ones(length(betas_all{2}(:,1)),1).*(2-0.15),betas_all{2}(:,1)','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None');

% linear model
scatter(ones(length(betas_all{1}(:,2)),1).*(1+0.15),betas_all{1}(:,2)','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None');
scatter(ones(length(betas_all{2}(:,2)),1).*(2+0.15),betas_all{2}(:,2)','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None');

plot([1-0.15; 1+0.15],[betas_all{1}(:,1)';betas_all{1}(:,2)'],'Color',[.8 .8 .8],'LineStyle','-');
plot([2-0.15; 2+0.15],[betas_all{2}(:,1)';betas_all{2}(:,2)'],'Color',[.8 .8 .8],'LineStyle','-');

eb = errorbar(1-0.15,mean(betas_all{1}(:,1)),sem(betas_all{1}(:,1),1),'LineWidth',2.5);
eb.Color = [0,0,0];
eb = errorbar(1+0.15,mean(betas_all{1}(:,2)),sem(betas_all{1}(:,2),1),'LineWidth',2.5);
eb.Color = [0,0,0];
eb = errorbar(2-0.15,mean(betas_all{2}(:,1)),sem(betas_all{2}(:,1),1),'LineWidth',2.5);
eb.Color = [0,0,0];
eb = errorbar(2+0.15,mean(betas_all{2}(:,2)),sem(betas_all{2}(:,2),1),'LineWidth',2.5);
eb.Color = [0,0,0];
pvals = [];
[~,pvals(1),~,s1] = ttest(betas_all{1}(:,1),betas_all{1}(:,2));
[~,pvals(2),~,s2] = ttest(betas_all{2}(:,1),betas_all{2}(:,2));
[~,pvals(3),~,s3] = ttest(betas_all{1}(:,1)-betas_all{1}(:,2),betas_all{2}(:,1)-betas_all{2}(:,2));

pvals
s1
effsize = compute_cohensD('t2',mean(betas_all{1}(:,1)),std(betas_all{1}(:,1),0,1),mean(betas_all{1}(:,2)),std(betas_all{1}(:,2),0,1))

s2
effsize = compute_cohensD('t2',mean(betas_all{2}(:,1)),std(betas_all{2}(:,1)),mean(betas_all{2}(:,2)),std(betas_all{2}(:,2)))

s3
effsize = compute_cohensD('t2',mean(betas_all{1}(:,1)-betas_all{1}(:,2)),std(betas_all{1}(:,1)-betas_all{1}(:,2)),mean(betas_all{2}(:,2)-betas_all{2}(:,2)),std(betas_all{2}(:,2)-betas_all{2}(:,2)))

sigstar({[0.9,1.1],[1.9,2.1],[1,2]},pvals)

box off;
set(gca,'XTick',1:length(betas_all));
set(gca,'XTickLabel',phaseName);
set(gca,'TickDir','out');
ylabel('\rm \beta estimate (a.u.)');
% ylim([-0.2,0.8]);
xlim([0,length(betas_all)+1]);
legend([b(1),b(2)],{'Factorised Model','Linear Model'},'Location','NorthEastOutside')
legend('boxoff');

end
