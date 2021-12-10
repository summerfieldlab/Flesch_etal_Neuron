function fmri_preproc_segmentation()
  %% fmri_preproc_segmentation
  %
  % segments structural into different tissue images
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  params = fmri_preproc_setParams();

  disp(['Tissue segmentation of structural']);
  for ii = 1:length(params.num.goodSubjects)
    subID = params.num.goodSubjects(ii);
    subjectDirName = set_fileName(subID);
    disp(['... job specification for subject : ', num2str(subID)]);

    cd([params.dir.imDir subjectDirName '/' params.dir.structSubDir]);

    % collect co-registered  structural image
    structDir = [params.dir.imDir subjectDirName '/' params.dir.structSubDir];
    structFile   = spm_select('List', structDir, params.regex.structCoregImages);

    matlabbatch{1}.spm.spatial.preproc = params.seg; % channel,tissue,warp
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {structFile};

    % save and run job
    save('batchFile_segmentStruct.mat','matlabbatch');

    disp(['... Tissue segmentation for subject ' num2str(subID)])
    spm_jobman('run','batchFile_segmentStruct.mat');

    clear matlabbatch
  end
