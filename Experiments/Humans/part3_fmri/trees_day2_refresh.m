function trees_day2_refresh()
	%% TREES_EXPT()
	%
	% main expt file
	%
	% Timo Flesch, 2018
	%
  outDir = [pwd, '/files/data/'];
  % treeDir = [pwd '/files/data_training/']
  
  addpath(genpath(pwd));
  data = struct();
	% request subject details
	subj = expt_prompt('train');
	save([outDir 's' num2str(subj.id) '_refresher_subData.mat'],'subj');

  % load unique tree indices
	trees = lib_loadMTurkTrees(subj.id);

  % comment out function below, now just used for debugging purposes
  % trees = lib_sampleSetsWithoutReplacement([1,12],50);

	%% EXPERIMENT
	data = struct();
	data = expt_runner(subj,trees{1},'refresher');
	save([outDir 's' num2str(subj.id) '_refresher_allData.mat'],'data');


end
