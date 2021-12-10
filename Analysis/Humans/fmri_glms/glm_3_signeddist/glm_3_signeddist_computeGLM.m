% performs voxel-wise (whole brain) univariate glm estimation of pre-defined conditions
% dmat contains conditions (convolved with hrf) and run-specific motion-parameters as nuisance regressors
% a separate set of parameters is estimated for each run
% a constant dummy variable (n columns) encodes the run ID and accounts for scanner drifts
% two-step process:
% 1. specify 1st level glm
% 2. estimate 1st level glm
%
% Timo Flesch, 2018,
% Human Information Processing Lab,
% Experimental Psychology Department
% University of Oxford

params = glm_3_signeddist_params();


for ii = 1:length(params.num.goodSubjects)
  subID = params.num.goodSubjects(ii);
  subjectDirName = set_fileName(subID);
  outDir_spec = [params.dir.glmDir subjectDirName '/' params.dir.dmatSubDir];
  outDir_est  = [params.dir.glmDir subjectDirName '/' params.dir.estSubDir];

  % move to output directory
  if ~exist(params.dir.glmDir,'dir')
    mkdir(params.dir.glmDir)
  end
  cd(params.dir.glmDir);
  if ~exist(subjectDirName,'dir')
    mkdir(subjectDirName);
    cd(subjectDirName);
    mkdir(params.dir.estSubDir);
    mkdir(params.dir.dmatSubDir);
    mkdir(params.dir.tSubDir);

    cd(params.dir.dmatSubDir);
  else
    cd(outDir_spec);
    delete('SPM.mat','*.nii','batchFile_spec.mat');
  end


  matlabbatch = {};
  % set parameters
  matlabbatch{1}.spm.stats.fmri_spec.dir            =        cellstr(outDir_spec);
  matlabbatch{1}.spm.stats.fmri_spec.timing.units   =           params.dmat.units;
  matlabbatch{1}.spm.stats.fmri_spec.timing.RT      =              params.dmat.TR;
  matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t  =   params.dmat.microtime_res;
  matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = params.dmat.microtime_onset;

  matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
  matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs =   params.misc.basisfuncts;
  matlabbatch{1}.spm.stats.fmri_spec.volt             =      params.misc.volterra;
  matlabbatch{1}.spm.stats.fmri_spec.global           =   params.misc.global_norm;
  matlabbatch{1}.spm.stats.fmri_spec.mthresh          =       params.misc.mthresh;
  matlabbatch{1}.spm.stats.fmri_spec.mask             = cellstr(params.misc.mask);
  matlabbatch{1}.spm.stats.fmri_spec.cvi              =    params.misc.serialcorr;


  if (subID==22) || (subID==14)
    runIDs = params.num.s22runIDs;
  else
    runIDs = params.num.runIDs;
  end
  for jj = 1:params.num.runs
      runID = runIDs(jj);
      % load EPIs
      funcDir = [params.dir.imDir subjectDirName '/' params.dir.runSubDir num2str(runID,'%04d') '/'];
      fileNames = spm_select('List', funcDir,params.regex.functional);
      matlabbatch{1}.spm.stats.fmri_spec.sess(jj).scans = cellstr([repmat(funcDir,size(fileNames,1),1) fileNames]);

      % load condition file....
      matlabbatch{1}.spm.stats.fmri_spec.sess(jj).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
      matlabbatch{1}.spm.stats.fmri_spec.sess(jj).multi = cellstr([params.dir.conditionDir 'conditions_' params.glmName '_sub' num2str(subID) '_run' num2str(jj) '.mat']);
      % ...  and motion regressors
      fileName = spm_select('List', funcDir,params.regex.motionregs);
      matlabbatch{1}.spm.stats.fmri_spec.sess(jj).regress = struct('name', {}, 'val', {});
      matlabbatch{1}.spm.stats.fmri_spec.sess(jj).multi_reg = cellstr([funcDir fileName]);

      % set high pass filter
      matlabbatch{1}.spm.stats.fmri_spec.sess(jj).hpf = params.misc.highpass;
  end

  % save batch
  cd(outDir_spec);
  save('batchFile_spec.mat','matlabbatch');

  % specify model
  cd(outDir_spec);
  disp(['Now specifying model for subject ' num2str(subID)]);
  spm_jobman('run','batchFile_spec.mat');
  clear matlabbatch;

  % if desired (highly recommended!), review design matrix before estimation begins
  if params.monitor.reviewDMAT
    cd(outDir_spec);
    load('SPM.mat');
    spm_DesRep('DesMtx',SPM.xX);
    spm_DesRep('DesOrth',SPM.xX)
  end

  % estimate parameters
  cd(outDir_est);
  clear matlabbatch;
  matlabbatch{1}.spm.stats.fmri_est.spmmat = cellstr({[outDir_spec 'SPM.mat']});
  save('batchFile_est.mat','matlabbatch');
  disp(['Now estimating model for subject ' num2str(subID)]);
  spm_jobman('run','batchFile_est.mat');
  clear matlabbatch;
end
