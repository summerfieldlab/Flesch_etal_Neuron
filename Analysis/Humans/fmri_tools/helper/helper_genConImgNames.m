function cNames = helper_genConImgNames(n);
%% CNAMES = HELPER_GENCONIMGNAMES(N)
%
% generates cell array with file names
%
% Timo Flesch, 2018
cNames = cell(n,1);
for (ii = 1:n)
  if (ii < 10)
    cNames{ii} = ['con_000' num2str(ii) '.nii'];
  else
    cNames{ii} = ['con_00' num2str(ii) '.nii'];
  end
end

end
