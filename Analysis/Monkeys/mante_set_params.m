function params = mante_set_params(projdir)
    %% MANTE_SET_PARAMS()
    %
    % a few parameters for analysis of mante data 
    %
    % Timo Flesch, 2020

    params = struct();
    % project folder structure
    params.dir = struct();
    params.dir.project = projdir ;
    params.dir.analysis  =  [params.dir.project 'analysis/'];
    params.dir.data      =      [params.dir.project 'data/'];
    params.dir.figures   =   [params.dir.project 'figures/'];
    params.dir.results   =   [params.dir.project 'results/'];
    

    % analysis
    params.analysis = struct();   
    params.analysis.n_monks = 2; 
    params.analysis.monknames = {'A','F'};
    % params for windowed analysis
    params.analysis.t_interval = 218:651; 
    % params for time series analysis
    params.analysis.t_window = 49; % 25 ms time window for averaging
    params.analysis.t_step = 50;
    params.analysis.trial_length = 751;   
    % params for binned analysis (early middle late)
    params.analysis.t_bins = {[1:217], [218:434],[435:651]};

    params.analysis.n_stimdir = 6;
    params.analysis.n_stimcol = 6;
    params.analysis.n_stimctx = 2;

    % stats
    params.stats = struct();
    params.stats.run_permtest = true;
    params.stats.n_perms = 1000;


    % mvpa 
    params.mvpa.n_samples = 20;
    