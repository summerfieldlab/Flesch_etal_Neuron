
/* **************************************************************************************

Shows raphael objects (opacity to 1)
(c) Timo Flesch, 2016 [timo.flesch@gmail.com]
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */
function beginTrial() {


    showCue();
    // step 5: show stimulus after timeout, wait for response
    coding.timestamp = getTimestamp();
    board.countdown.stimTimeIn  = setTimeout(showTrial, parameters.stimulus_timein);
    startCountdown(); // manages response time
}

function showTrial() {
  /*
    shows a trial
  */

  // allow answering w arrow keys, disable space bar
  coding.answering = true;
  coding.newblock  = false;
  showKeys();
  showStimuli();

}


function showCue() {
  /*
    shwos contextual cue (orchard)
  */

  showFixation(board.fixation);

  // board.cue.object.attr({"opacity":1});
  board.cue.context.attr({"opacity":1});
  // board.cue.text.attr({"opacity":1});

}


function showAction(whichAction) {
/*
  whichAction: [1/0] (plant,don't plant)
*/
  if (whichAction) {
    showAccept();
  }
  else {
    showReject();
  }
}


function showAccept() {
/*
  shows that subject planted tree
*/
  // show AcceptKey()
  showChoiceRect(1);
  // if training, provide feedback
  if ((sdata.expt_phase[coding.index] ==1 || sdata.expt_phase[coding.index] ==3)  && coding.task == 0) {
    setTimeout(showFeedbackPos,parameters.feedback_timein);
  }
}


function showReject() {
/*
  shows that subject did not plant tree
*/

  showChoiceRect(0);
  // if training, provide feedback (same holds as earlier)
  if ((sdata.expt_phase[coding.index] ==1 || sdata.expt_phase[coding.index] ==3)  && coding.task == 0) {
    setTimeout(showFeedbackNeg,parameters.feedback_timein);
  }
}


function showStimuli() {
 /*
  stimulus presentation interval - shows stim in front of blurred cue
 */
 hideFixation(board.fixation);
  board.stimuli.tree.attr({"opacity": 1});
}


function showKeys() {
 /*
   shows instruction keys
 */
  board.instructions.keys[0].attr({"opacity": 1});
  board.instructions.keys[1].attr({"opacity": 1});
}



// function showFeedback() {
// /*
//   shows feedback (reward + resized tree)
// */
//   if(sdata.resp_category[coding.index]){ // if decided to plant
//    showFeedbackPos();
//   } else { // if not decided to plant or no response
//    showFeedbackNeg();
//    // setTimeout(showFeedbackNeg,parameters.feedback_timein);
//
//   }
// }


function showChoiceRect(acceptOrReject) {
  /*
  acceptOrReject: [1/0];
  */
  switch (acceptOrReject) {
    case 1:
      // square around chosen val
      if (sdata.expt_keyassignment[coding.index]) {
        board.leftfeedback.rect.attr({"opacity": 1});
      }
      else {
        board.rightfeedback.rect.attr({"opacity": 1});
      }
      break;
    case 0:
      // square around chosen val
      if (sdata.expt_keyassignment[coding.index]) {
         board.rightfeedback.rect.attr({"opacity": 1});
      }
      else {
        board.leftfeedback.rect.attr({"opacity": 1});
      }
      break;
  }


}

function showFeedbackPos() {
/*
  displays positive feedback (subject decided to plant tree)
*/
  hideStimuli();
  hideKeys();
  showFixation(board.fixation);
  // updateFeedback();
  board.leftfeedback.object.attr({"opacity": 1});
  board.rightfeedback.object.attr({"opacity": 1});
  // square around chosen val
  if (sdata.expt_keyassignment[coding.index]) {
    board.leftfeedback.rect.attr({"opacity": 1});
  }
  else {
    board.rightfeedback.rect.attr({"opacity": 1});
  }
}
function showFeedbackNeg() {
 /*
  displays negative feedback (subject decided not to plant tree)
 */
  hideStimuli();
  hideKeys();
  showFixation(board.fixation);
  // updateFeedback();
  board.leftfeedback.object.attr({"opacity": 1});
  board.rightfeedback.object.attr({"opacity": 1});
  // square around chosen val
  if (sdata.expt_keyassignment[coding.index]) {
     board.rightfeedback.rect.attr({"opacity": 1});
  }
  else {
    board.leftfeedback.rect.attr({"opacity": 1});
  }


}


function showBreakMessage() {
 /*
  displays block break message
 */

  // if it's a regular break, just display break text. otherwise, display reminder
  // that context is going to switch (e.g. when in phase parameters.nb_mainTrainingPhaseID (3), blocked group and context(n) != context(n-1)
  board.block = {};
  board.block.centre = board.paper.centre;
  if (parameters.task_id.slice(0,1) == 'blocked' & coding.phase==3 && ((sdata.expt_contextIDX[coding.index] != sdata.expt_contextIDX[coding.index-1]))) {
  board.block.text = parameters.txt.break_taskSwitch;
  }
  else {
    board.block.text = parameters.txt.break_taskStay;
  }
  board.block.object = drawText(board.paper.object,board.block.centre,board.block.text);
  board.block.object.attr({"font-size": board.font_medsize});
  coding.newblock  = true;
}

function showStartMessage() {
  /*
    displays training session instructions
  */

  board.block = {};
  board.block.centre = board.paper.centre;
  board.block.text = parameters.txt.startMessage;

  board.block.object = drawText(board.paper.object,board.block.centre,board.block.text);
  board.block.object.attr({"font-size": board.font_medsize});

}
