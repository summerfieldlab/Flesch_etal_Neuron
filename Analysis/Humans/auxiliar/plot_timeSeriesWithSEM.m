function pHandle = plot_timeSeriesWithSEM(yData,colVals,figureIndex)
%
% plots mean data plus SEM,
% returns plot handle
%
% author: Timo Flesch
% summerfield lab, experimental psychology department, university of oxford
%

pHandle = cell(4,1);

%% compute group mean and group SEM
yMean = 				    nanmean(yData,1);
ySEM  = std(yData,0,1)./sqrt(size(yData,1));

%% plot mean and shaded SEM
figure(figureIndex); set(gcf,'Color','w');
hold on;

% mean of data:
f_yMean = plot(yMean,'LineStyle','-','LineWidth',2,'Color',colVals); 
hold on;

% sem as area:
f_ySEMarea = fill([1:size(yMean,2) fliplr(1:size(yMean,2))],[yMean+ySEM fliplr(yMean-ySEM)],colVals);
set(f_ySEMarea,'FaceAlpha',0.4);
set(f_ySEMarea,'EdgeColor',[1 1 1]);
set(f_ySEMarea,'EdgeAlpha',0.4);

% baseline:
plot(get(gca,'XLim'),[0,0],'k-');
plot([0,0],get(gca,'YLim'),'k-');
hold on;

% return figure handles:
pHandle{1,1} = 	  f_yMean;
pHandle{2,1} = f_ySEMarea;
