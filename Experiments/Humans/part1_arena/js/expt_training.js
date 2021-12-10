
function startTraining() {
  if (!startedexperiment) { return; }
  // hide stuff
  hideInstructions();
  hideFeedback();
  hideDots();
  // launch training
  updateTraining();
  setTimeout(interTraining,parameters.training_wait);
}

function interTraining() {
  coding.training.wait = false;
}

function handleSpace() {
  if (coding.training.wait) { return; }
  coding.training.index++;
  setTimeout(interTraining,parameters.training_wait);
  updateTraining();
  coding.training.wait = true;
}

function updateTraining() {
  if(coding.training.finished) { return; }

  // demo functions
  var nullfunction = function() {}
  var instr_func_2 = function() {
    var prototype = genPrototype();
    for (var id=0; id<board.stimulus.number; id++) { board.stimulus.dots[id].object.attr({"x": prototype.x[id],"y": prototype.y[id]}); }
    showDots();
  }
  var instr_func_6 = function() { showFeedbackPos(); }
  var instr_func_7 = function() { hideFeedback(); showFeedbackNeg(); }
  var instr_func_8 = function() { hideFeedback(); startCountdown(); }
  var instr_fun_10 = function() { stopCountdown(); hideDots(); hideBackground(); }
  var lastfunction = function() { board.training.object.remove(); showBackground(); showInstructions(); coding.training.finished = true; newTrial(); }

  // instructions
  var instructions = [];
  instructions[ 0] = { func: nullfunction, text: "Hello! We will first explain you the task. Please press SPACE to move forward." };
  instructions[ 1] = { func: nullfunction, text: "In this experiment, you will complete many trials.\nWe will require you to give a response on every trial." };
  instructions[ 2] = { func: instr_func_2, text: "On each trial, you will see a set of dots like this one." };
  instructions[ 3] = { func: instr_func_2, text: "Or this one.\nYou will see many of these. One on each trial." };
  instructions[ 4] = { func: instr_func_2, text: "These are just examples.\n(They don't match the ones you will actually see in the task you're about to do.)" };
  instructions[ 5] = { func: nullfunction, text: "Each set of dots belongs exclusively to one out of two categories.\nYour task is to learn these two categories and decide to which category each set of dots belongs to." };
  instructions[ 6] = { func: nullfunction, text: "Once you respond, you will receive informative feedback." };
  instructions[ 7] = { func: instr_func_6, text: "You will see this green text when you respond correctly." };
  instructions[ 8] = { func: instr_func_7, text: "You will see this red text when your response is not correct." };
  instructions[ 9] = { func: instr_func_8, text: "If you take too long to respond, a countdown will pop-up." };
  instructions[10] = { func: nullfunction, text: "If the countdown gets to its end before you give a response,\nthe experiment will be cancelled." };
  instructions[11] = { func: nullfunction, text: "You will respond with the arrow keys LEFT and RIGHT.\nEach of them will correspond to one or the other categories." };
  instructions[12] = { func: instr_fun_10, text: "Ready to start? Good luck!" };
  instructions[13] = { func: lastfunction, text: "" };
  
  // apply instructions
  board.training.object.attr({"text": instructions[coding.training.index].text});
  instructions[coding.training.index].func();
}

