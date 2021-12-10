% computes voxel-wise contrasts and t-value map at single subject level
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
  outDir_con  = [params.dir.glmDir subjectDirName '/' params.dir.tSubDir];
  % move to output directory
  cd(outDir_con);

  % generate contrast vectors
  c = glm_3_signeddist_gencontrast();

  % setup contrast batch
  for cIDX = 1:length(c.T.labels)
    matlabbatch{1}.spm.stats.con.consess{cIDX}.tcon.name    =    c.T.labels{cIDX};
    matlabbatch{1}.spm.stats.con.consess{cIDX}.tcon.convec  = c.T.vectors(cIDX,:);
    matlabbatch{1}.spm.stats.con.consess{cIDX}.tcon.sessrep =              'none';
  end

  matlabbatch{1}.spm.stats.con.delete = params.files.overwriteContrasts;
  matlabbatch{1}.spm.stats.con.spmmat = cellstr({[outDir_spec 'SPM.mat']});

  % save batch
  cd(outDir_con);
  save('batchFile_con.mat','matlabbatch');

  % if desired, review contrasts
  if params.monitor.reviewContrasts
    helper_dispContrastVectors(c,params);
  end
  disp(['Now computing contrasts for subject ' num2str(subID)]);
  spm_jobman('run','batchFile_con.mat');
  clear matlabbatch;
end
