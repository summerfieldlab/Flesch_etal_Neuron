function fmri_preproc_dicomImport()
  %% FMR_PREPROC_DICOMIMPORT()
  %
  % imports DICOM images as NIFTI files
  % functionals: 3d (one image per tr)
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford


  params = fmri_preproc_setParams();

  disp(['Converting DICOM images to NIFTI images']);
  for ii = 1:length(params.num.goodSubjects)
    subID = params.num.goodSubjects(ii);
    subjectDirName = set_fileName(subID);
    disp(['... job specification for subject : ', num2str(subID)]);

    % cd so that .mat and .ps files are written in functional dir
    % cd([params.dir.imDir subjectDirName '/' params.dir.epiSubDir]);
    cd([params.dir.imDir subjectDirName '/']);

    allEPIfiles = [];
    % collect all EPIs (of all sessions)
    for runID = 1:params.num.runs
      % funcDir = [params.dir.imDir subjectDirName '/' params.dir.epiSubDir  params.dir.runSubDir num2str(runID) '/'];
      funcDir = [params.dir.imDir subjectDirName '/' params.dir.runSubDir num2str(runID) '/'];
      fileNames   = spm_select('List', funcDir, '.*.dcm$');
      runFiles    = cellstr([repmat(funcDir,size(fileNames,1),1) fileNames]);
      allEPIfiles = [allEPIfiles; runFiles];  % add files of all sessions
      fileNames   =                      [];
      runFiles    =                      [];
    end
    
    % collect raw structural image
    structDir = [params.dir.imDir subjectDirName '/' params.dir.structSubDir];
    % structDir = [params.dir.imDir subjectDirName '/'];
    clear fileNames;
    fileNames   = spm_select('List', params.dir.structSubDir, '.*.dcm$');
    structFiles    = cellstr([repmat(structDir,size(fileNames,1),1) fileNames]);
    % set parameters
    matlabbatch{1}.spm.util.import.dicom = params.import.dicom;

    % add images
    matlabbatch{1}.spm.util.import.dicom.data = [allEPIfiles;structFiles];

    % output directory
    params.import.dicom.outdir = [ params.dir.imDir subjectDirName];

    % save and run job
    save('batchFile_dicom2nifti.mat','matlabbatch');

    disp(['... converting EPIs and Structural for subject ' num2str(subID)])
    spm_jobman('run','batchFile_dicom2nifti.mat');
    clear matlabbatch
  end


end
