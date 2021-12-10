function conditions = conditions_scan(params)
	%% CONDITIONS_PRE_SCAN(PARAMS)
	%
	% generates vectors with trial indices for all
	% experimental variables
	%
	% Timo Flesch, 2018

	conditions = struct();
	%% set up exp conditions
	% context/task idces: north =1 , south =2
	if (params.task.order == 1)
		conditions.task         =  repmat([1.*ones(params.num.trials_run/2,1);2.*ones(params.num.trials_run/2,1)],params.num.runs,1);
	elseif (params.task.order == 2)
		conditions.task         =  repmat([2.*ones(params.num.trials_run/2,1);1.*ones(params.num.trials_run/2,1)],params.num.runs,1);
	end

	% feature levels
	[branchiness,leafiness] =          meshgrid(params.num.branchiness,params.num.leafiness);
	conditions.branchiness  = repmat(branchiness(:),params.num.exemplars*params.num.tasks,1);
	conditions.leafiness    =   repmat(leafiness(:),params.num.exemplars*params.num.tasks,1);

	% rewards
	[rewBranch,rewLeaf]     =   meshgrid(params.num.rewards_branchiness,params.num.rewards_leafiness);
	rewBranch               = repmat(rewBranch(:),params.num.exemplars*params.num.tasks,1);
	rewLeaf                 =   repmat(rewLeaf(:),params.num.exemplars*params.num.tasks,1);

	% category labels & actual rewards
	conditions.rewards(conditions.task==1)                 = rewLeaf(conditions.task==1);
	conditions.rewards(conditions.task==2)                 = rewBranch(conditions.task==2);
	conditions.rewards                                     = conditions.rewards';
	conditions.categories                                  = conditions.rewards > 0;

	% exemplars, shuffle across all runs
	tmp1 = [repmat(params.task.exemplars,1,params.num.trials_total/params.num.exemplars)];
	ii = randperm(params.num.trials_total);
	conditions.exemplars = tmp1(ii)';

	% tree IDs (=filenames)
	for ii = 1:length(conditions.exemplars)
		% disp(['B' num2str(conditions.branchiness(ii)) 'L' num2str(conditions.leafiness(ii)) '_' conditions.exemplars{ii}]);
		conditions.treeIDs{ii} = ['B' num2str(conditions.branchiness(ii)) 'L' num2str(conditions.leafiness(ii)) '_' num2str(conditions.exemplars(ii))];
	end

	runIDs  = [helper_makeRunIDs(params)]';

	% shuffle everything (within runs!)
	for ii = 1:params.num.runs
		shuffIdces = randperm(params.num.trials_run);
		fields = fieldnames(conditions);
		for jj = 1:numel(fields)
			data = conditions.(fields{jj})(runIDs==ii);
			conditions.(fields{jj})(runIDs==ii) = data(shuffIdces);
		end
	end


	%% add counters and idces
	% counters
	conditions.counter_total  = [1:params.num.trials_total]';
	conditions.counter_run  = [helper_makeRunCounter(params)]';
	% idces
	conditions.sess         = params.task.sessID.*ones(params.num.trials_total,1);
	conditions.runs         = runIDs;

	%% add jitters
	if params.timing.doJitter
		conditions.ITIs = [];
		for ii = 1:params.num.runs
			conditions.ITIs = [conditions.ITIs; lib_jitter(params.timing.minITI,params.timing.maxITI,params.num.trials_run)];
		end
	else
		conditions.ITIs = repmat(params.timing.ITI,params.num.trials_total,1);
	end


end

function counter_run = helper_makeRunCounter(params)
	counter_run = [];
	for ii =1:params.num.runs
		counter_run = [counter_run,1:params.num.trials_run];
	end
end

function ids_run = helper_makeRunIDs(params)
	ids_run = [];
	for ii =1:params.num.runs
		ids_run = [ids_run,ii.*ones(1,params.num.trials_run)];
	end
end
