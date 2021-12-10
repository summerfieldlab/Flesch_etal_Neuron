
function addPaper() {
  // BOARD
  // fonts
  board.font_bigsize   = 40;
  board.font_medsize   = 15;
  board.font_tinysize  = 12;
  // paper (paper)
  board.paper = {};
  board.paper.width  = window.innerWidth;
  board.paper.height = window.innerHeight;
  board.paper.centre = [0.5*window.innerWidth , 0.5*window.innerHeight];
  board.paper.rect   = [
                        0,
                        0,
                        board.paper.width,
                        board.paper.height
                       ];
  board.paper.object = drawPaper(board.paper.rect);

  // FIXATION
  board.fixation = createFixation(board.paper);

  // CUE

  board.cue        =                 {};
  board.blurcue    =                 {};
  board.garden     =                 {};

  board.cue.centre = board.paper.centre;
  drawGarden("north");

  hideCue();


  // KEYS
  // instructions (text)
  board.instructions = {};

  // board.instructions.keys         =  drawKeys(parameters.keyassignment);
  board.instructions.keys         =  drawKeys(sdata.expt_keyassignment[coding.index]);

  hideKeys();



  //STIMULUS

  board.stimuli = {};

  board.stimuli.tree = drawTree("B5L4_1.png");
  hideStimuli();

  // CLOCK
  // drawClock([board.paper.centre[0], board.paper.centre[1]+150]);

  // COUNTDOWN
  drawCountdown("#808",1);
  board.countdown.total = 5;

  // FEEDBACK
  // leftfeedback (text)
  board.leftfeedback = {};
  board.leftfeedback.centre   = board.paper.centre;
  board.leftfeedback.text     = "+0";
  board.leftfeedback.colour   = parameters.visuals.cols.fbn_neu;
  board.leftfeedback.object   = drawText(board.paper.object,[board.leftfeedback.centre[0]-50,board.leftfeedback.centre[1]-160],board.leftfeedback.text);
  board.leftfeedback.object.attr({"font-size": board.font_bigsize});
  board.leftfeedback.object.attr({"fill": board.leftfeedback.colour});
  // leftfeedback (rect) indicates selected option
  board.leftfeedback.rect = drawChoiceRect('left');
  // board.leftfeedback.rect = drawRect(board.paper.object,[board.paper.centre[0]-90,board.paper.centre[1]-160,80,80])
  // board.leftfeedback.rect.attr({"stroke-width":4,fill:"grey"})
  // rightfeedback (text)
  board.rightfeedback = {};
  board.rightfeedback.centre   = board.paper.centre;
  board.rightfeedback.text     = "+0";
  board.rightfeedback.colour   = parameters.visuals.cols.fbn_neu;
  board.rightfeedback.object   = drawText(board.paper.object,[board.leftfeedback.centre[0]+50,board.leftfeedback.centre[1]-160],board.rightfeedback.text);
  board.rightfeedback.object.attr({"font-size": board.font_bigsize});
  board.rightfeedback.object.attr({"fill": board.rightfeedback.colour});
  // rightfeedback (rect) indicates selected option
  board.rightfeedback.rect = drawChoiceRect('right');
  // board.rightfeedback.rect = drawRect(board.paper.object,[board.paper.centre[0]+12,board.paper.centre[1]-200,80,80])
  // board.rightfeedback.rect.attr({"stroke-width":4,fill:"grey"});

  hideFeedback();



}
