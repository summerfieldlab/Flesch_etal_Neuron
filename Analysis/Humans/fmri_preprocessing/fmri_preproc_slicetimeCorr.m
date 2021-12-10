function fmri_preproc_slicetimeCorr()
  %% fmri_preproc_slicetimeCorr
  %
  % performs slice time correction on EPIs
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  params = fmri_preproc_setParams();

  disp(['Slice Time Correction of EPIs']);
  for ii = 1:length(params.num.goodSubjects)
    subID = params.num.goodSubjects(ii);
    subjectDirName = set_fileName(subID);

    disp(['... job specification for subject : ', num2str(subID)]);


    cd([params.dir.imDir subjectDirName '/']);

    % collect all EPIs (of all sessions)
    if (subID==22) || (subID==14)
      runIDs = params.num.s22runIDs;
    else
      runIDs = params.num.runIDs;
    end
    for jj = 1:params.num.runs
        runID = runIDs(jj);
        funcDir = [params.dir.imDir subjectDirName '/' params.dir.runSubDir num2str(runID,'%04d') '/'];
        % select realigned and unwarped EPI images
        fileNames   = spm_select('List', funcDir,['^' params.reuw.unwarpresl.prefix  'f.*\.nii$']);
        runFiles = cellstr([repmat(funcDir,size(fileNames,1),1) fileNames]);

        matlabbatch{1}.spm.temporal.st.scans{jj} = runFiles;
        fileNames = [];
        runFiles  = [];
    end
    % populate batch fields with options from params file:
    fns = fieldnames(params.st);
    for jj = 1:length(fns)
      matlabbatch{1}.spm.temporal.st.(fns{jj}) = params.st.(fns{jj});
    end
    % save job description
    save('batchFile_stEPI.mat','matlabbatch');

    % run job
    disp(['... slice-time correct EPIs for subject ' num2str(subID)])    
    spm_jobman('run','batchFile_stEPI.mat');
    clear matlabbatch
  end
