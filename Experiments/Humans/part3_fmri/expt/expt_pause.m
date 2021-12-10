function expt_pause(w,txtCol,pauseKey)
  %% EXPT_PAUSE(W,TXTCOL,PAUSEKEY)
  %
  % pauses the experiment
  % and waits for button press
  %
  % Timo Flesch, 2018


  DrawFormattedText(w,'The experiment is paused. Press ''p'' to continue...','center','center',txtCol);
  continueTask = false;
  Screen(w,'Flip');

  while (~continueTask)
    [~,~,keyCode] = KbCheck();
    if (keyCode(pauseKey))
      continueTask = true;
    end
  end
  return;
end
