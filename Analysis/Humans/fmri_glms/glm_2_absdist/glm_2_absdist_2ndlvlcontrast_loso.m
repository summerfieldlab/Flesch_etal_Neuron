% computes voxel-wise t-tests at group level
% in a leave one subject out approach.
% generates one subfolder per subject for each contrast.
%
% this script is used for LOSO functional ROI estimation
%
% Timo Flesch, 2018,
% Human Information Processing Lab,
% Experimental Psychology Department
% University of Oxford

% parameters and file names
params = glm_2_absdist_params();
c = glm_2_absdist_gencontrast();
cNames = helper_genConImgNames(length(c.T.labels));

% directories
outDir_group = [params.dir.glmDir params.dir.losoSubDir];
cd(params.dir.glmDir);

if ~exist('params.dir.losoSubDir', 'dir')
    mkdir(params.dir.losoSubDir);
    cd(params.dir.losoSubDir);
else
    cd(outDir_group);
end

% iterate through contrasts
for cIDX = 1:length(c.T.labels)
    cd(outDir_group);

    if ~exist(['con_' num2str(cIDX) '/'], 'dir')
        mkdir(['con_' num2str(cIDX) '/']);
        cd(['con_' num2str(cIDX) '/']);
    else
        cd(['con_' num2str(cIDX) '/']);
    end

    disp(['now estimating contrast ' num2str(cIDX)]);
    % iterate through folds
    for fIDX = 1:params.num.subjects
        cd(outDir_group);
        cd(['con_' num2str(cIDX) '/']);

        disp(['fold ' num2str(fIDX)]);
        % kick out "active" subject
        subVect = 1:params.num.subjects;
        subVect(fIDX) = [];
        % make subdir for current fold
        if ~exist(['fold_' num2str(fIDX) '/'], 'dir')
            mkdir(['fold_' num2str(fIDX) '/']);
            cd(['fold_' num2str(fIDX) '/']);
        else
            cd(['fold_' num2str(fIDX) '/']);
        end

        outDir_group_con_fold = [outDir_group ['con_' num2str(cIDX) '/' 'fold_' num2str(fIDX) '/']];

        matlabbatch = {};
        % iterate through all n-1 subjects
        iiSub = 1;

        for subIDX = subVect
            subjectDirName = set_fileName(subIDX);
            outDir_con = [params.dir.glmDir subjectDirName '/' params.dir.tSubDir];
            % move to directory that contains all contrast images
            cd(outDir_con);
            matlabbatch{1}.spm.stats.factorial_design.des.t1.scans{iiSub, 1} = [outDir_con cNames{cIDX}];
            iiSub = iiSub + 1;
        end

        cd(outDir_group_con_fold);

        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = [];
        % matlabbatch{1}.spm.stats.factorial_design.masking.im             =    0;
        % matlabbatch{1}.spm.stats.factorial_design.masking.em             = {[]};
        matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = [];
        matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = [];
        matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
        matlabbatch{1}.spm.stats.factorial_design.dir = {outDir_group_con_fold};

        % save batch
        save('batchFile_spec2.mat', 'matlabbatch');

        % specify model
        disp(['Now specifying 2nd-level model for contrast ' num2str(cIDX) ': ' c.T.labels{cIDX}]);
        spm_jobman('run', 'batchFile_spec2.mat');
        clear matlabbatch;

        % if desired (highly recommended!), review design matrix before estimation begins
        if params.monitor.reviewDMAT
            cd(outDir_group_con_fold);
            load('SPM.mat');
            spm_DesRep('DesMtx', SPM.xX);
            spm_DesRep('DesOrth', SPM.xX)
        end

        % estimate parameters
        cd(outDir_group_con_fold);
        matlabbatch = {};
        matlabbatch{1}.spm.stats{1}.fmri_est.spmmat = {[outDir_group_con_fold 'SPM.mat']};
        save('batchFile_est2.mat', 'matlabbatch');
        disp(['Now estimating 2nd-level model for contrast ' num2str(cIDX) ': ' c.T.labels{cIDX}]);
        spm_jobman('run', 'batchFile_est2.mat');
        clear matlabbatch;

        % specify contrasts (2nd level rfx)
        matlabbatch = {};
        matlabbatch{1}.spm.stats.con.spmmat = {[outDir_group_con_fold 'SPM.mat']};
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = c.T.labels{cIDX};
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.convec = [1];
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
        matlabbatch{1}.spm.stats.con.delete = params.files.overwriteContrasts;

        save('batchFile_con2.mat', 'matlabbatch');
        disp(['2nd level test for 1st level contrast ' num2str(cIDX) ': ' c.T.labels{cIDX}]);
        spm_jobman('run', 'batchFile_con2.mat');
        clear matlabbatch;
    end

end
