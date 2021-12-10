function pHandle = aux_ts_shadederrorbar(xData,yData,colVals,figureIndex,errorStyle,alphaLevel,lineStyle)
%
% plots mean data plus SEM,
% returns plot handle
%
% author: Timo Flesch, 2016
% summerfield lab, experimental psychology department, university of oxford
%
if ~exist('alphaLevel')
  alphaLevel = 0.4;
end

if ~exist('lineStyle')
  lineStyle = '-';
end

if ~exist('errorStyle')
    errorStyle = 'sem';
end

pHandle = cell(4,1);

%% compute group mean and group SEM
yMean = 				    mean(yData,1);

if strcmp(errorStyle,'sem')
    yErr  = std(yData,0,1)./sqrt(size(yData,1));
elseif strcmp(errorStyle,'std')
    yErr  = std(yData,0,1);
elseif strcmp(errorStyle,'2std')
    yErr  = 2*std(yData,0,1);
end

%% plot mean and shaded SEM
figure(figureIndex); set(gcf,'Color','w');
hold on;

% mean of data:
f_yMean = plot(xData,yMean,'LineStyle',lineStyle,'LineWidth',2,'Color',colVals);
hold on;

% sem as area:
f_ySEMarea = fill([xData fliplr(xData)],[yMean+yErr fliplr(yMean-yErr)],colVals);
set(f_ySEMarea,'FaceAlpha',alphaLevel);
set(f_ySEMarea,'EdgeColor',[1 1 1]);
set(f_ySEMarea,'EdgeAlpha',alphaLevel);

% baseline:
plot(get(gca,'XLim'),[0,0],'k-');
plot([0,0],get(gca,'YLim'),'k-');
hold on;

% return figure handles:
pHandle{1,1} = 	  f_yMean;
pHandle{2,1} = f_ySEMarea;

end