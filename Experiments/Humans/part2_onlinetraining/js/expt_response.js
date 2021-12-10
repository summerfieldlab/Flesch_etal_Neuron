
// <!-- Response methods -->
function handleLeft()  {
  if(sdata.expt_keyassignment[coding.index]) {
    handleResponse('Left', 1);
  }
  else {
    handleResponse('Left', 0);
  }
}


function handleRight() {
 if(sdata.expt_keyassignment[coding.index]) {
    handleResponse('Right', 0);
  }
  else {
    handleResponse('Right', 1);
  }
}

function handleResponse(key,category) {
  if(coding.answering){
    if(isnan(category)) {
      handleNoResponse();
    }
    coding.answering = false;
    stopCountdown();
    clearTimeout(board.countdown.stimTimeOut); // this is important, otherwise, stim might disappear during next trial..
    saveResponse(key,category);
    // hideStimuli();


    // show choice:
    showAction(category);

    if(sdata.resp_correct[coding.index]){
      setTimeout(nextTrial,parameters.feedpos_timeout);
    } else {
      setTimeout(nextTrial,parameters.feedneg_timeout);
    }

  }
}

function handleNoResponse() {

  sdata.resp_timestamp[coding.index]         = NaN;
  sdata.resp_reactiontime[coding.index]      = NaN;
  sdata.resp_category[coding.index]          = NaN;
  sdata.resp_correct[coding.index]           = NaN;
  sdata.resp_reward[coding.index]            =   0; // didn't plant tree

  if(!isFullscreen() && startedexperiment && !finishedexperiment) {
//    finishExperiment_noresponse();
  }
}

function saveResponse(key,category) {

  sdata.resp_timestamp[coding.index]         = getTimestamp();
  sdata.resp_reactiontime[coding.index]      = getSecs(coding.timestamp);
  sdata.resp_category[coding.index]          = category;
  sdata.resp_correct[coding.index]           = bin2num((sdata.expt_catIDX[coding.index]>=0)==sdata.resp_category[coding.index]);
  sdata.resp_reward[coding.index]            = sdata.expt_rewardIDX[coding.index]*category;
  sdata.resp_return[coding.index]            = (coding.trial) ? (sum(sdata.resp_reward.slice(coding.return,coding.index+1))) : (0); //cumsum within current block up to current trial.
  // save reward & return for optimal policy
  sdata.expt_rewardOPT[coding.index]         = sdata.expt_rewardIDX[coding.index]*bin2num(sdata.expt_catIDX[coding.index]>0);
  sdata.expt_returnOPT[coding.index]          = (coding.trial) ? (sum(sdata.expt_rewardOPT.slice(coding.return,coding.index+1))) : (0);

  //debugging:
  if (true) {
    console.log("**********************")
    console.log('INDEX ' +                        coding.index);
    console.log('TRIAL ' +                        coding.trial);
    console.log('PHASE ' +                        coding.phase);
    console.log('Phase_ID' +     sdata.expt_phase[coding.index]);
    console.log('Block_ID' +     sdata.expt_block[coding.index]);
    console.log('task '  + sdata.expt_contextIDX[coding.index]);
    console.log(treeName);
    console.log("Stimulus Category (good or bad tree): "   +   sdata.expt_catIDX[coding.index]);
    console.log("Response Category (plant or not plant): " + sdata.resp_category[coding.index]);
    console.log('\n');
    console.log("Stimulus Reward: " + sdata.expt_rewardIDX[coding.index]);
    console.log("Optimal Reward: "  + sdata.expt_rewardOPT[coding.index]);
    console.log("Response Reward: " +    sdata.resp_reward[coding.index]);
    console.log('\n');
    console.log("Optimal Return: "  + sdata.expt_returnOPT[coding.index]);
    console.log("Response Return: " +    sdata.resp_return[coding.index]);
    console.log('\n');
    console.log("Response Correct:" +   sdata.resp_correct[coding.index]);
  }
}
