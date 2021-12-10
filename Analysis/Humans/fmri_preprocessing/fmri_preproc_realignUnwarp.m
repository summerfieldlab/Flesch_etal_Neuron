function fmri_preproc_realignUnwarp()
  %% fmri_preproc_realignUnwarp
  %
  % realigns and unwarps functional EPIs
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  params = fmri_preproc_setParams();

  disp(['Realignment and Unwarping of functional EPIs']);
  for ii = 1:length(params.num.goodSubjects)
    subID = params.num.goodSubjects(ii);
    subjectDirName = set_fileName(subID);

    disp(['... job specification for subject : ', num2str(subID)]);


    cd([params.dir.imDir subjectDirName '/']);

    allEPIfiles = [];
    % collect all EPIs (of all sessions)
    if (subID==22) || (subID==14)
      runIDs = params.num.s22runIDs;
    else
      runIDs = params.num.runIDs;
    end
    for jj = 1:params.num.runs
        runID = runIDs(jj);
        funcDir = [params.dir.imDir subjectDirName '/' params.dir.runSubDir num2str(runID,'%04d') '/'];
        % select raw EPI images
        fileNames   = spm_select('List', funcDir,'^f.*\.nii$');
        runFiles = cellstr([repmat(funcDir,size(fileNames,1),1) fileNames]);

        matlabbatch{1}.spm.spatial.realignunwarp.data(jj).scans = runFiles;
        matlabbatch{1}.spm.spatial.realignunwarp.data(jj).pmscan = '';
        fileNames = [];
        runFiles  = [];
    end

    % populate batch fields with options from params file:
    matlabbatch{1}.spm.spatial.realignunwarp.eoptions   = params.reuw.estimate;
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions = params.reuw.unwarpest;
    matlabbatch{1}.spm.spatial.realignunwarp.uwroptions = params.reuw.unwarpresl;

    % save job description
    save('batchFile_reuwEPI.mat','matlabbatch');

    % run job
    disp(['... realigning and unwarping EPIs for subject ' num2str(subID)])
    spm_jobman('run','batchFile_reuwEPI.mat');
    clear matlabbatch
  end
