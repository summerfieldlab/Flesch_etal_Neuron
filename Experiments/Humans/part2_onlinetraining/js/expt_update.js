
// <!-- Update methods -->


function updateCue () {

  if(sdata.expt_contextIDX[coding.index]==1){
      gardenName = "north";
    }
  else if (sdata.expt_contextIDX[coding.index]==2){
      gardenName = "south";
    }
  drawGarden(gardenName);

}


function updateStimuli() {
 treeName = 'B'.concat(sdata.expt_branchIDX[coding.index].toString()).concat('L').concat(sdata.expt_leafIDX[coding.index].toString()).concat('_'+sdata.expt_exemplarIDX[coding.index].toString().concat('.png'));
  board.stimuli.tree = drawTree(treeName);
}


function updateKeys() {
    board.instructions.keys         =  drawKeys(sdata.expt_keyassignment[coding.index]);
}


// display correct reward values
function updateFeedback() {

board.rightfeedback.rect = drawChoiceRect('right');
board.leftfeedback.rect  = drawChoiceRect('left');

if (sdata.expt_keyassignment[coding.index]) {

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
  board.leftfeedback.object   = drawText(board.paper.object,[board.paper.centre[0]-parameters.visuals.size.stim[0]/2-parameters.visuals.size.keyIMG[0]+40,board.paper.centre[1]],board.leftfeedback.text);
  board.leftfeedback.object.attr({"font-size": board.font_bigsize});
  board.leftfeedback.object.attr({"fill": board.leftfeedback.colour,"stroke":board.leftfeedback.colour, "stroke-width":"1px", "paint-order":"stroke"});

  board.rightfeedback.text     = "+0";
  board.rightfeedback.colour   = parameters.visuals.cols.fbn_neu;
  board.rightfeedback.object   = drawText(board.paper.object,[board.paper.centre[0]+parameters.visuals.size.stim[0]/2+40,board.paper.centre[1]],board.rightfeedback.text);
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

  board.rightfeedback.object   = drawText(board.paper.object,[board.paper.centre[0]+parameters.visuals.size.stim[0]/2+40,board.paper.centre[1]],board.rightfeedback.text);
  board.rightfeedback.object.attr({"font-size": board.font_bigsize});
  board.rightfeedback.object.attr({"fill": board.rightfeedback.colour,"stroke":board.rightfeedback.colour, "stroke-width":"1px", "paint-order":"stroke"});

  board.leftfeedback.text     = "+0";
  board.leftfeedback.colour   = parameters.visuals.cols.fbn_neu;
  board.leftfeedback.object   = drawText(board.paper.object,[board.paper.centre[0]-parameters.visuals.size.stim[0]/2-parameters.visuals.size.keyIMG[0]+40,board.paper.centre[1]],board.leftfeedback.text);
  board.leftfeedback.object.attr({"font-size": board.font_bigsize});
  board.leftfeedback.object.attr({"fill": board.leftfeedback.colour,"stroke":board.leftfeedback.colour, "stroke-width":"1px", "paint-order":"stroke"});
}
//NOTE where did that come from? spaghetti code, eh?
//hideFeedback();
}
