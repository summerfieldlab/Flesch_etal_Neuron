function fmri_preproc_coregistration()
  %% fmri_preproc_slicetimeCorr
  %
  % coregisters structural with mean EPI
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  params = fmri_preproc_setParams();

  disp(['Coregistration of structural with mean functional of first run']);
  for ii = 1:length(params.num.goodSubjects)
    subID = params.num.goodSubjects(ii);
    subjectDirName = set_fileName(subID);
    if (subID==22) || (subID==14)
      runIDs = params.num.s22runIDs;
    else
      runIDs = params.num.runIDs;
    end

    disp(['... job specification for subject : ', num2str(subID)]);

    % cd so that .mat and .ps files are written in functional dir
    cd([params.dir.imDir subjectDirName '/']);

    % populate batch fields with options from params file:
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions = params.coreg.estimate;

    % reference image: mean functional from first run
    funcDir = [params.dir.imDir subjectDirName '/' params.dir.runSubDir num2str(runIDs(1),'%04d') '/'];
    meanFile   = spm_select('List', funcDir, ['mean' params.reuw.unwarpresl.prefix 'f.*\.nii$']);
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {[funcDir meanFile]};

    % source image (the one to coregister with the ref): structural
    structDir = [params.dir.imDir subjectDirName '/' params.dir.structSubDir];
    structFile   = spm_select('List', params.dir.structSubDir, params.regex.structRawImages);
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {[structDir structFile]};
    % no other images to coregister:
    matlabbatch{1}.spm.spatial.coreg.estwrite.other = {''};

    % save job description
    save('batchFile_coreg.mat','matlabbatch');


    % run job
    disp([' coregistering structural with mean functional for subject ' num2str(subID)])
    spm_jobman('run','batchFile_coreg.mat');
    clear matlabbatch
  end
