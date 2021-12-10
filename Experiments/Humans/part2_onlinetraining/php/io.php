<!-- **************************************************************************************

Writes JSON Files to folder on server
(c) Timo Flesch, 2016/17 
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** -->


<?php
   $json = $_POST['json'];

   $file = fopen('../data/final/dissimratings.json','w+');
  fwrite($file, $json);
  fclose($file);
 
?>