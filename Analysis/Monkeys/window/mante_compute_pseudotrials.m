function [X_train, X_test, y_train, y_test] = mante_compute_pseudotrials(params)
    %% mante_compute_pseudotrials()
    % computes pseudotrials for each condition

    % import data
    monk = 1;
    cd([params.dir.data 'monkey' num2str(monk) '/']);
    allunits = dir('*.mat');
    unitstruct = struct();
    disp('loading data')

    for d = 1:length(allunits)
        load(allunits(d).name);
        unitstruct(d).unit = unit;
    end

    disp('...done');

    % get feature values
    unit = unitstruct(1).unit;
    ud = unique(unit.task_variable.stim_dir);
    uc = unique(unit.task_variable.stim_col);
    cc = unique(unit.task_variable.context);

    X_train = [];
    y_train = [];
    X_test = [];
    y_test = [];

    for u1 = 1:length(ud)

        for u2 = 1:length(uc)

            for c = 1:length(cc)
                X_train_d = [];
                X_test_d = [];
                y_train_d = [];
                y_test_d = [];
                % sample pseudo-trials:
                for d = 1:length(unitstruct)
                    % obtain trial indices:
                    indx = find(unit.task_variable.stim_dir == ud(u1) & unit.task_variable.stim_col == uc(u2) & unit.task_variable.context == cc(c));
                    % get corresponding unit activity,divided into training and test, and average across time:
                    unitact_train = nanmean(unit.response(indx(1:round(length(indx) / 2)), params.analysis.t_interval), 2);
                    unitact_test = nanmean(unit.response(indx(round(length(indx) / 2):end), params.analysis.t_interval), 2);
                    % sample trials
                    trials_train = datasample(unitact_train, params.mvpa.n_samples, 1);
                    trials_test = datasample(unitact_test, params.mvpa.n_samples, 1);
                    X_train_d(:,d) = trials_train;
                    X_test_d(:,d) = trials_test;
                   
                end
                y_train_d = repmat([c,u1,u2],[params.mvpa.n_samples,1]);
                y_test_d = repmat([c,u1,u2],[params.mvpa.n_samples,1]);

                X_train = [X_train; X_train_d];
                X_test = [X_test; X_test_d];
                y_train = [y_train; y_train_d];
                y_test = [y_test; y_test_d];
            end

        end

    end

end
