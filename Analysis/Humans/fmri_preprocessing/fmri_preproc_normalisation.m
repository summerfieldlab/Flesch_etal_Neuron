function fmri_preproc_normalisation()
  %% fmri_preproc_normalisation
  %
  % normalises images from native to MNI space
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  params = fmri_preproc_setParams();

  disp(['Normalising EPIs and Structural from native to mni space']);
  for ii = 1:length(params.num.goodSubjects)
    subID = params.num.goodSubjects(ii);
    subjectDirName = set_fileName(subID);
    disp(['... job specification for subject : ', num2str(subID)]);

    % cd so that .mat and .ps files are written in functional dir
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
        % select unwarped and st-corrected images (with which structural is coregistered)
        fileNames   = spm_select('List', funcDir, params.regex.epiCoregImages);
        runFiles = cellstr([repmat(funcDir,size(fileNames,1),1) fileNames]);
        allEPIfiles = [allEPIfiles; runFiles];  % add files of all sessions
        fileNames = [];
        runFiles  = [];
    end

    % collect co-registered structural image
    structDir = [params.dir.imDir subjectDirName '/' params.dir.structSubDir];
    structFile   = spm_select('List', params.dir.structSubDir, params.regex.structCoregImages);

    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {[structDir structFile]};
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = allEPIfiles;

    % import parameters
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions = params.norm.estimate;
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions =    params.norm.write;
    % save and run job
    save('batchFile_normEPIStruct.mat','matlabbatch');

    disp(['... normalising EPIs and Structural for subject ' num2str(subID)])
    spm_jobman('run','batchFile_normEPIStruct.mat');
    clear matlabbatch
  end
