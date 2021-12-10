function data = expt_runner(subj,trees,sess)
	%
	% runs experiments
	% Timo Flesch, 2018

	% (1) execute selected session
	switch sess
	case 'refresher' % refresher (training)
		params  =    params_refresher(subj,trees);
		expt    =    conditions_refresher(params);
		results =     task_refresher(params,expt);

	case 'scan' % test (scan)
		params  = params_scan(subj,trees);
		expt    = conditions_scan(params);
		results =  task_scan(params,expt);
	end

	data.params     =  params;
	data.conditions =    expt;
	data.results    = results;
	data.subj       =    subj;
	data.trees      =   trees;

end
