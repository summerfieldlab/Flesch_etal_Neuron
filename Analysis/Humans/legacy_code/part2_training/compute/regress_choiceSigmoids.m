function results = regress_choiceSigmoids(choiceProbs)
	%% REGRESS_CHOICESIGMOIDS(RESULTS)
	%
	% fits sigmoids to choice probabilities
	% for all  phases, and both the relevant as well as
	% irrelevant dimensions at single subject label
	% and performs inferential stats on estimated slopes (signrank against 0)
	%
	% Timo Flesch, 2018
	% Human Information Processing Lab, Experimental Psychology Department
	% University of Oxford

	allSigmas = struct();
	allStats  = struct();

	rewVals = [-2:2];
	dimNames = {'rel','irrel'};
	expPhases  = fieldnames(choiceProbs);

	for ii =1:length(expPhases)
		for dd = 1:length(dimNames)
			for subj = 1:size(choiceProbs.p1.rel.choice,1)
				y_bl = squeeze(choiceProbs.(expPhases{ii}).(dimNames{dd}).choice(subj,:));
				% disp('noop')
				% fit sigmoids
				sigmas = fitSigmoid([rewVals',y_bl']);
				allSigmas.(expPhases{ii}).(dimNames{dd})(subj,:) = sigmas;
			end
			% perform within-group test for statistical significance
			for bb = 1 :size(allSigmas.(expPhases{ii}).(dimNames{dd}),2)
				[p,~,s] = signrank(squeeze(allSigmas.(expPhases{ii}).(dimNames{dd})(:,bb)),0,'method','approximate');
				allStats.(expPhases{ii}).(dimNames{dd}).p(bb) =      p;
				allStats.(expPhases{ii}).(dimNames{dd}).z(bb) = s.zval;
			end
		end
	end
	results.sigmas = allSigmas;
	results.stats  =  allStats;
end

function betas = fitSigmoid(data)
betas = nlinfit(data(:,1),data(:,2),@mysigmoid,[.1]);
end

function fun = mysigmoid(b,x)
fun = 1./(1+exp(-b(1)*(x)));
end
