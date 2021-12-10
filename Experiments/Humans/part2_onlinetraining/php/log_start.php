<html>
  <head>
    <title>
      log_start.php
    </title>
  </head>
  <body>
<?php
  // log path
  $LOG_PATH = '../docs/log.txt';

  // save POST variables
  $ID   = $_POST["id"];
  $TASK = $_POST["task"];
  
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
  $LOG_TEXT =file_get_contents($LOG_PATH);

  // check POST variables & update log
  if (empty($TASK))   {
    $LOG_TEXT .= $DATE." : ERROR. 'task' not specified\n";
    echo "      empty 'id'   <br /> \n"; 
  } elseif (empty($ID)) {
    $LOG_TEXT .= $DATE." : ".$TASK." : ERROR. 'id' not specified\n";
    echo "      empty 'task' <br /> \n";
  } else {
    $LOG_TEXT .= $DATE." : ".$TASK." : starting participant '".$ID."'\n";
    echo "    log '".$LOG_PATH."' has been updated. <br />\n";
  }

  // write log
  file_put_contents($LOG_PATH,$LOG_TEXT);
  chmod($LOG_PATH, 0777);
?>    
  </body>
</html>

