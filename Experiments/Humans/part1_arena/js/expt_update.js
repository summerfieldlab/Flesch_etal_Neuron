
<!-- Update methods -->


function updateCue () {

if(sdata.expt_contextIDX[coding.index]==1){
    board.cue.object.remove();
    gardenName = "orchard_north.png"

    drawGarden(gardenName,0);
    drawGarden(gardenName,1);
  }
else if (sdata.expt_contextIDX[coding.index]==2){
    board.cue.object.remove();
    gardenName ="orchard_south.png"

    drawGarden(gardenName,0);
    drawGarden(gardenName,1);
  }

}

function updateStimuli() {
  board.stimuli.context.remove();
   board.stimuli.context = drawRect(board.paper.object,[board.paper.centre[0]-120,board.paper.centre[1]-120,240,240]);
  board.stimuli.context.attr({stroke:"black","stroke-width":5});
  board.stimuli.context.attr({fill:"grey"});
 treeName = 'B'.concat(sdata.expt_branchIDX[coding.index].toString()).concat('L').concat(sdata.expt_leafIDX[coding.index].toString()).concat('_'+sdata.expt_exemplarIDX[coding.index].concat('.png'));
  board.stimuli.tree = drawTree(treeName);
  hideStimuli();
  removeKeys();
  board.instructions.keys         =  drawKeys(parameters.keyassignment);
  hideKeys();

}

// display correct reward values
function updateFeedback() {
removeFeedback();
board.rightfeedback.rect = drawRect(board.paper.object,[board.paper.centre[0]+12,board.paper.centre[1]-180,80,80])
board.rightfeedback.rect.attr({"stroke-width":4});
board.leftfeedback.rect = drawRect(board.paper.object,[board.paper.centre[0]-90,board.paper.centre[1]-180,80,80])
board.leftfeedback.rect.attr({"stroke-width":4});

if (parameters.keyassignment) {

  if(sdata.expt_rewardIDX[coding.index]>0){
    board.leftfeedback.text     = "+".concat(sdata.expt_rewardIDX[coding.index].toString());
    board.leftfeedback.colour   = parameters.visuals.cols.fbn_pos;
  }
  else if(sdata.expt_rewardIDX[coding.index]<0){
   board.leftfeedback.text     = (sdata.expt_rewardIDX[coding.index].toString());
   board.leftfeedback.colour   = parameters.visuals.cols.fbn_neg;
  }
  else if(sdata.expt_rewardIDX[coding.index]==0){
    board.leftfeedback.text    = "+".concat(sdata.expt_rewardIDX[coding.index].toString());
    board.leftfeedback.colour  = parameters.visuals.cols.fbn_neu;
  }
  board.leftfeedback.object   = drawText(board.paper.object,[board.leftfeedback.centre[0]-50,board.leftfeedback.centre[1]-140],board.leftfeedback.text);
  board.leftfeedback.object.attr({"font-size": board.font_bigsize});
  board.leftfeedback.object.attr({"fill": board.leftfeedback.colour,"stroke":board.leftfeedback.colour, "stroke-width":"1px", "paint-order":"stroke"});



  board.rightfeedback.text     = "+0";
  board.rightfeedback.colour   = parameters.visuals.cols.fbn_neu;
  board.rightfeedback.object   = drawText(board.paper.object,[board.leftfeedback.centre[0]+50,board.leftfeedback.centre[1]-140],board.rightfeedback.text);
  board.rightfeedback.object.attr({"font-size": board.font_bigsize});
  board.rightfeedback.object.attr({"fill": board.rightfeedback.colour,"stroke":board.rightfeedback.colour, "stroke-width":"1px", "paint-order":"stroke"});
}

else {

  if(sdata.expt_rewardIDX[coding.index]>0){
    board.rightfeedback.text     = "+".concat(sdata.expt_rewardIDX[coding.index].toString());
    board.rightfeedback.colour   = parameters.visuals.cols.fbn_pos;
  }

  else if(sdata.expt_rewardIDX[coding.index]<0){
   board.rightfeedback.text     = (sdata.expt_rewardIDX[coding.index].toString());
   board.rightfeedback.colour   = parameters.visuals.cols.fbn_neg;
  }
  else if(sdata.expt_rewardIDX[coding.index]==0){
    board.rightfeedback.text    = "+".concat(sdata.expt_rewardIDX[coding.index].toString());
    board.rightfeedback.colour  = parameters.visuals.cols.fbn_neu;
  }

  board.rightfeedback.object   = drawText(board.paper.object,[board.rightfeedback.centre[0]+50,board.rightfeedback.centre[1]-140],board.rightfeedback.text);
  board.rightfeedback.object.attr({"font-size": board.font_bigsize});
  board.rightfeedback.object.attr({"fill": board.rightfeedback.colour,"stroke":board.rightfeedback.colour, "stroke-width":"1px", "paint-order":"stroke"});

  board.leftfeedback.text     = "+0";
  board.leftfeedback.colour   = parameters.visuals.cols.fbn_neu;
  board.leftfeedback.object   = drawText(board.paper.object,[board.leftfeedback.centre[0]-50,board.leftfeedback.centre[1]-140],board.leftfeedback.text);
  board.leftfeedback.object.attr({"font-size": board.font_bigsize});
  board.leftfeedback.object.attr({"fill": board.leftfeedback.colour,"stroke":board.leftfeedback.colour, "stroke-width":"1px", "paint-order":"stroke"});
}
hideFeedback();
}
