function conditions = conditions_refresher(params)
	%% CONDITIONS_PRE_behav(PARAMS)
	%
	% generates vectors with trial indices for all
	% experimental variables
	%
	% Timo Flesch, 2018

	conditions = struct();
	%% set up exp conditions
	% context/task idces: north =1 , south =2
	if (params.task.order == 1)
		conditions.task         =  [1.*ones(params.num.trials_total/2,1);2.*ones(params.num.trials_total/2,1)];
	elseif (params.task.order == 2)
		conditions.task         =  [2.*ones(params.num.trials_total/2,1);1.*ones(params.num.trials_total/2,1)];
	end

	% feature levels
	[branchiness,leafiness] =          meshgrid(params.num.branchiness,params.num.leafiness);
	conditions.branchiness  = repmat(branchiness(:),params.num.exemplars*params.num.tasks,1);
	conditions.leafiness    =   repmat(leafiness(:),params.num.exemplars*params.num.tasks,1);

	% rewards
	[rewBranch,rewLeaf]     =              meshgrid(params.num.rewards_branchiness,params.num.rewards_leafiness);
	rewBranch               = repmat(rewBranch(:),params.num.exemplars*params.num.tasks,1);
	rewLeaf                 =   repmat(rewLeaf(:),params.num.exemplars*params.num.tasks,1);

	% category labels & actual rewards
	conditions.rewards(conditions.task==1)                 = rewLeaf(conditions.task==1);
	conditions.rewards(conditions.task==2)                 = rewBranch(conditions.task==2);
	conditions.rewards                                     = conditions.rewards';
	conditions.categories                                  = conditions.rewards > 0;

	% exemplars
	tmp1 = [repmat(params.task.exemplars,1,params.num.trials_total/params.num.exemplars)];
	ii = randperm(params.num.trials_total);
	conditions.exemplars = tmp1(ii)';

	% tree IDs (=filenames)
	for ii = 1:length(conditions.exemplars)
		conditions.treeIDs{ii} = ['B' num2str(conditions.branchiness(ii)) 'L' num2str(conditions.leafiness(ii)) '_' num2str(conditions.exemplars(ii))];
	end

	% shuffle everything!
	shuffIdces = randperm(params.num.trials_total);
	fields = fieldnames(conditions);
	for ii = 1:numel(fields)
		conditions.(fields{ii}) = conditions.(fields{ii})(shuffIdces);
	end

	%% add counters
	% counters
	conditions.counter_total  = [1:params.num.trials_total]';
	conditions.counter_block  = [helper_makeBlockCounter(params)]';
	% session idces
	conditions.sess         = params.task.sessID.*ones(params.num.trials_total,1);


	%% add jitters (naive, not balanced across conditions)
	if params.timing.doJitter
		conditions.ITIs = lib_jitter(params.timing.minITI,params.timing.maxITI,params.num.trials_total);
	else
		conditions.ITIs = repmat(params.timing.ITI,params.num.trials_total,1);
	end
end

function counter_block = helper_makeBlockCounter(params)
	counter_block = [];
	for ii =1:params.num.blocks
		counter_block = [counter_block,1:params.num.trials_block];
	end
end
