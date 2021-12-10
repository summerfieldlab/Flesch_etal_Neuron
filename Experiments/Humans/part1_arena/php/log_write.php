<html>
  <head>
    <title>
      log_write.php
    </title>
  </head>
  <body>
<?php
  // log path
  $LOG_PATH = '../docs/log.txt';

  // save POST variables
  $ID   = $_POST["id"];
  $TASK = $_POST["task"];
  $PATH = $_POST["path"];
  $MOVE = $_POST["move"];

  // date variable
  date_default_timezone_set('Europe/London');
  $DATE = date("Y/m/d H:i:s");

  // create log if does not exist
  if (!file_exists($LOG_PATH)) {
    $LOG_FILE = fopen($LOG_PATH, "w+");
    fclose($LOG_FILE);
    echo "    log '".$LOG_PATH."' created <br /> \n";
  }

  // read log
  $LOG_TEXT = file_get_contents($LOG_PATH);

  // subject variables
  $SUB_PATH = "../".$PATH."/".$ID.".txt";
  $SUB_MOVE = "../".$MOVE."/".$ID.".txt";
  $SUB_TEXT = "";

  // check POST variables & update log
  if (empty($TASK))   {
    $LOG_TEXT .= $DATE." : error. no task specified\n";
    echo "      empty 'id'   <br /> \n"; 
  } elseif (empty($ID)) {
    $LOG_TEXT .= $DATE." : ".$TASK." : error. no id specified\n";
    echo "      empty 'task' <br /> \n";
  } elseif (empty($PATH)) {
    $LOG_TEXT .= $DATE." : ".$TASK." : error. no path specified\n";
    echo "      empty 'task' <br /> \n";
  } else {
    // update subject file
    foreach($_POST as $key => $value) {
      if(strcmp($key,"id")  ==0) { continue; }
      if(strcmp($key,"task")==0) { continue; }
      if(strcmp($key,"path")==0) { continue; }
      if(strcmp($key,"move")==0) { continue; }
      $SUB_TEXT .= $key." : ".$value."\n";
    }
    $LOG_TEXT .= $DATE." : ".$TASK." : writing participant '".$ID."' in '".$SUB_PATH."'\n";
    file_put_contents($SUB_PATH,$SUB_TEXT);
    chmod($SUB_PATH, 0777);

    // move subject file
    if (!empty($MOVE)) {
      rename($SUB_PATH,$SUB_MOVE);
      $LOG_TEXT .= $DATE." : ".$TASK." : moving file from '".$SUB_PATH."' to '".$SUB_MOVE."'\n";
    }
  }

  // write log
  file_put_contents($LOG_PATH,$LOG_TEXT);
  chmod($LOG_PATH, 0777);
?>    
  </body>
</html>

