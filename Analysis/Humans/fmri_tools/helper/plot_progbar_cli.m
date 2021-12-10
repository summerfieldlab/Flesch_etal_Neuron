function revStr = plot_progbar_cli(idx,maxValue,revStr)
  %% plot_progbar_cli()
  %
  % plots the process of a process
  % in percent inside the console
  % and avoids output on muliple lines
  %
  % Usage: within a for loop, call it like this
  % revStr = '';
  % for ii = 1:maxVal
  %   revStr = plot_progbar_cli(ii,maxVal,revStr);
  % end
  %
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford


  percentFinished = 100*idx/maxValue;
  progStr = sprintf('Percent completed: %3.1f', percentFinished);
  fprintf([revStr, progStr]);
  revStr = repmat(sprintf('\b'), 1, length(progStr));
  if idx==maxValue
    fprintf('\n');
    revStr = '';
  end
end
