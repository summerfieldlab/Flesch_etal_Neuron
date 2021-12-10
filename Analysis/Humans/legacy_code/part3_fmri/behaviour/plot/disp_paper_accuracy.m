function disp_paper_accuracy(acc_day1,acc_scan)
%% disp_accuracy(acc_day1,acc_refresher,acc_scan)
%
% displays mean accuracies for each phase
%
% Timo Flesch, 2018,
% Human Information Processing Lab,
% Experimental Psychology Department
% University of Oxford

badsubs = [19,28]; % no training data available (performed at 81 and 64 % in scanner)

sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));


phaseName      = {'test_{baseline}', 'test_{scan}'};


% all data ---------------------------------------------------------------------------------------------------------------------
acc_all = {};
% acc_all{1} = acc_day1.complete.p1;
% acc_all{2} = acc_day1.complete.p2;
% acc_all{3} = acc_day1.complete.p3;
% acc_all{4} = acc_refresher.complete;
% acc_all{5} = acc_scan.complete;

acc_all{1} = acc_day1.complete.p2;
acc_all{2} = acc_scan.complete;
acc_all{2}([badsubs])  = [];
colvals = linspace(0.4,0.6,length(acc_all));

figure();set(gcf,'Color','w');
for ii = 1:length(acc_all)
  b = bar(ii,mean(acc_all{ii}));
  b.FaceColor = [1,1,1].*colvals(ii);
  hold on;
  scatter(ones(length(acc_all{ii}),1).*ii,acc_all{ii}','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None')
  eb = errorbar(ii,mean(acc_all{ii}),sem(acc_all{ii},2),'LineWidth',2.5);
  eb.Color = [0,0,0];
end
plot([1;2],[acc_all{1};acc_all{2}],'Color',[.8 .8 .8],'LineStyle','-');
[~,pval,~,s] = ttest(acc_all{1},acc_all{2});
pval
s
sigstar([1,2],pval);
box off;
set(gca,'XTick',1:length(acc_all));
set(gca,'XTickLabel',phaseName);
set(gca,'TickDir','out');
ylabel('\bf Accuracy [%]');
set(gca,'YTick',[0:.2:1])
set(gca,'YTickLabel',[0:20:100])
ylim([0.3,1.2]);
plot(get(gca,'XLim'),[.5 .5],'k--');
xlim([0,length(acc_all)+1]);
set(gcf,'Position', [2401, 323, 281, 420]);
%
% %% task switch -------------------------------------------------------------------------------------------------------------------
acc_taskSwitch = {};
acc_taskSwitch{1} = [acc_day1.taskSwitchStay.p2.stay,acc_day1.taskSwitchStay.p2.switch];
acc_taskSwitch{2} = [acc_scan.taskSwitchStay.stay,acc_scan.taskSwitchStay.switch];
acc_taskSwitch{2}(badsubs,:)  = [];

colvals = linspace(0.4,0.6,2);

figure();set(gcf,'Color','w');
e = [sem(acc_taskSwitch{1},1);sem(acc_taskSwitch{2},1)];
m = [mean(acc_taskSwitch{1},1);mean(acc_taskSwitch{2},1)];
b = barwitherr(e,m);
b(1).FaceColor = [1,1,1].*colvals(1);
b(2).FaceColor = [1,1,1].*colvals(2);
hold on;
% stay trials
% plotSpread_hack([acc_taskSwitch{1}(:,1),acc_taskSwitch{2}(:,1)],'distributionMarkers','o','xValues',[1,2],'xOffset',-0.15)
scatter(ones(length(acc_taskSwitch{1}(:,1)),1).*(1-0.15),acc_taskSwitch{1}(:,1)','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None')
scatter(ones(length(acc_taskSwitch{2}(:,1)),1).*(2-0.15),acc_taskSwitch{2}(:,1)','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None')

% switch trials
% plotSpread_hack([acc_taskSwitch{1}(:,2),acc_taskSwitch{2}(:,2)],'distributionMarkers','o','xValues',[1,2],'xOffset',+0.15)
scatter(ones(length(acc_taskSwitch{1}(:,2)),1).*(1+0.15),acc_taskSwitch{1}(:,2)','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None')
scatter(ones(length(acc_taskSwitch{2}(:,2)),1).*(2+0.15),acc_taskSwitch{2}(:,2)','MarkerFaceColor',[.8,.8,.8],'MarkerEdgeColor','None')

plot([1-0.15; 1+0.15],[acc_taskSwitch{1}(:,1)';acc_taskSwitch{1}(:,2)'],'Color',[.8 .8 .8],'LineStyle','-');
plot([2-0.15; 2+0.15],[acc_taskSwitch{2}(:,1)';acc_taskSwitch{2}(:,2)'],'Color',[.8 .8 .8],'LineStyle','-');
eb = errorbar(1-0.15,mean(acc_taskSwitch{1}(:,1)),sem(acc_taskSwitch{1}(:,1),1),'LineWidth',2.5);
eb.Color = [0,0,0];
eb = errorbar(1+0.15,mean(acc_taskSwitch{1}(:,2)),sem(acc_taskSwitch{1}(:,2),1),'LineWidth',2.5);
eb.Color = [0,0,0];
eb = errorbar(2-0.15,mean(acc_taskSwitch{2}(:,1)),sem(acc_taskSwitch{2}(:,1),1),'LineWidth',2.5);
eb.Color = [0,0,0];
eb = errorbar(2+0.15,mean(acc_taskSwitch{2}(:,2)),sem(acc_taskSwitch{2}(:,2),1),'LineWidth',2.5);
eb.Color = [0,0,0];
pvals = [];
[~,pvals(1),~,s1] = ttest(acc_taskSwitch{1}(:,1),acc_taskSwitch{1}(:,2));
[~,pvals(2),~,s2] = ttest(acc_taskSwitch{2}(:,1),acc_taskSwitch{2}(:,2));
[~,pvals(3),~,s3] = ttest(acc_taskSwitch{1}(:,1)-acc_taskSwitch{1}(:,2),acc_taskSwitch{2}(:,1)-acc_taskSwitch{2}(:,2));

pvals
s1
s2
s3
sigstar({[0.9,1.1],[1.9,2.1],[1,2]},pvals)
box off;
set(gca,'XTick',1:length(acc_taskSwitch));
set(gca,'XTickLabel',phaseName);
set(gca,'TickDir','out');
ylabel('\bf Accuracy [%]');
set(gca,'YTick',[0:.2:1])
set(gca,'YTickLabel',[0:20:100])
ylim([0.3,1.2]);
plot(get(gca,'XLim'),[.5 .5],'k--');
xlim([0,length(acc_taskSwitch)+1]);
legend([b(1),b(2)],{'Task Stay','Task Switch'},'Location','NorthEastOutside')
legend('boxoff');
set(gcf,'Position',[680   546   503   425]);

%% task switch vs stay - differences --------------------------------------------------------------------------------
figure();set(gcf,'Color','w');
