function trees_expt()
	%% TREES_EXPT()
	%
	% main expt file
	%
	% Timo Flesch, 2018
	%
  outDir = [pwd, '/files/data/'];
  addpath(genpath(pwd));
  data = struct();
  % request subject details
  subj = expt_prompt('scan');
  save([outDir 's' num2str(subj.id) '_scan_subData.mat'],'subj');

  % load unique tree indices
  trees = lib_loadMTurkTrees(subj.id);

  % comment out function below, now just used for debugging purposes
  % trees = lib_sampleSetsWithoutReplacement([1,12],50);


  %% EXPERIMENTS
  data = struct();
  data = expt_runner(subj,trees{2},'scan');
  save([outDir 's' num2str(subj.id) '_scan_allData.mat'],'data');


end
