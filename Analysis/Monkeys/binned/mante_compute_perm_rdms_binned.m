function results = mante_compute_perm_rdms_binned(params)
    %% mante_compute_perm_rdms_binned(params)
    % 
    % computes rdms from randomly permuted data 
    % to create a null distribution of rdms 
    %
    % labels are permuted at single neuron level 
    %
    % Timo Flesch, 2020
   
    
    results = struct();
    
    for monk = 1:params.analysis.n_monks
        % load data, (do this once, ow n_perms * n_units read ops.......)
        cd([params.dir.data 'monkey' num2str(monk) '/']);
        allunits = dir('*.mat');
        unitstruct = struct();
        disp('loading data')
        for d = 1:length(allunits)
            load(allunits(d).name);
            unitstruct(d).unit = unit;
        end
        % init rdm set         
        results(monk).rdms = nan(params.stats.n_perms,length(params.analysis.t_bins),params.analysis.n_stimdir^2*2,params.analysis.n_stimdir^2*2);
        % rAndOmiZeeee
        disp('starting permutation test')
        for perm = 1:params.stats.n_perms;
            tic
            rdms = NaN(length(params.analysis.t_bins),params.analysis.n_stimdir^2*2,params.analysis.n_stimcol^2*2);            
               
            for (ib = 1:length(params.analysis.t_bins))
                respmat = helper_compute_respmat(unitstruct,params,params.analysis.t_bins{ib},true);            
                rdms(ib,:,:) =  squareform(pdist(transpose(respmat(:,:))));                   
            end            

            if mod(perm,10)==0
                disp(['finished ' num2str(perm) '/' num2str(params.stats.n_perms) ' permutations']);
            end
            toc
            results(monk).rdms(perm,:,:,:) = rdms;
        end
        results(monk).name = ['monkey ' params.analysis.monknames{monk}];
    end
    cd(params.dir.results)
    save('monkey_perm_rdms_binned.mat','results');
    cd(params.dir.project);
    
end

