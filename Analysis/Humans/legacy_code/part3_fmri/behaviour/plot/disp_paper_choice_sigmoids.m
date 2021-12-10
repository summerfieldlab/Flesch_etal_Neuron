function allSigmas = disp_paper_choice_sigmoids(results_day1,results_scan)
	%
	%
	% plots group level choice probabilities together with
	% best fitting sigmoidal curves (just for illustration purposes fit on group level)
	%
	% Timo Flesch, 2018

	sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));

	rewVals = [-2:2];
	% dimNames = {'rel'};
	dimNames = {'rel','irrel'};
	styles   = {'-','-.'};
	colvals = linspace(0.3,0.7,2);
	% phaseLabels = {'Training 1'};
  	allSigmans = struct();

	phaseLabels = {'test_{baseline}', 'test_{scan}'};
	choiceProbs = struct();
	choiceProbs(1).rel = results_day1.p2.rel.choice;
	choiceProbs(1).irrel = results_day1.p2.irrel.choice;
	choiceProbs(2).rel = results_scan.rel.choice;
	choiceProbs(2).irrel = results_scan.irrel.choice;

	figure(); set(gcf,'Color','w');
	idx = 1
	for ii =1:length(phaseLabels)
		for dd = 1:length(dimNames)
			% plot errorbars of choice probas - blocked
			y_bl = squeeze(mean(choiceProbs(ii).(dimNames{dd}),1));
			err_bl = squeeze(sem(choiceProbs(ii).(dimNames{dd}),1));
			eb(1) = errorbar(rewVals,y_bl,err_bl,'o','MarkerSize',4,'MarkerEdgeColor',[1,1,1].*colvals(ii),'MarkerFaceColor',[1,1,1].*colvals(ii),'Color',[1,1,1].*colvals(ii));
			hold on;

			% plot group level sigmoidal fits - blocked
			warning off;
			sigmas = fitSigmoid([rewVals',y_bl']);
			allSigmas(ii).(dimNames{dd}).blocked = sigmas;
			x = -2:0.1:2;
			y_bl_hat = mysigmoid(sigmas,x);
			ph(idx) = plot(x,y_bl_hat,'Color',[1,1,1].*colvals(ii),'LineWidth',2,'LineStyle',styles{dd});
			idx = idx+1;
		end

	end
	legend([ph(1) ph(2) ph(3) ph(4)],{'Baseline - Relevant','Baseline - Irrelevant','Scan - Relevant','Scan - Irrelevant'},'Box','on','Location','NorthEastOutside');
	xlabel('Reward');
	ylabel('P(Plant)');
	set(gca,'XTick',-2:1:2);
	set(gca,'XTickLabel',[-50,-25,0,25,50]);
	ylim([0,1]);
	set(gca,'YTick',0:0.25:1);
	set(gca,'TickDir','out');
	box off;
	% set(gcf,'Position',[ 680   751   346   227]);
	set(gca,'LineWidth',1);
	title({'Choice Probabilities'});


end
function betas = fitSigmoid(data)

betas = nlinfit(data(:,1),data(:,2),@mysigmoid,[.1,.1,.1]);
% betas = nlinfit(data(:,1),data(:,2),@mysigmoid,[.1]);

end

function fun = mysigmoid(b,x)

fun = b(3) + (1-b(3)*2)./(1+exp(-b(1)*(x-b(2))));
% fun = 1./(1+exp(-b(1)*(x)));


end
