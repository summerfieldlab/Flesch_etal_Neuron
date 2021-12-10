
<!-- Hide methods -->
function hideTrial() {
  hideFixation(board.fixation);
  hideStimuli();
  // hideInstructions();
  // hideClock();
}

function hideCue() {
  // board.cue.object.attr({"opacity":0});
  board.cue.context.attr({"opacity":0});
  // board.cue.text.attr({"opacity":0});
}

function hideStimuli() {
  board.stimuli.tree.attr({"opacity": 0});
}


function hideKeys() {
  board.instructions.keys[0].attr({"opacity": 0});
  board.instructions.keys[1].attr({"opacity": 0});
}


function hideFeedback() {
  board.leftfeedback.object.attr({"opacity": 0});
  board.rightfeedback.object.attr({"opacity": 0});
  board.leftfeedback.rect.attr({"opacity": 0});
  board.rightfeedback.rect.attr({"opacity": 0});
}

function hideBlock() {
  board.block.object.remove(); // actually removeBlock (remove text)
}
