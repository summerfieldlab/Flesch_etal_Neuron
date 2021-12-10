
<div class="screenBox">

  <!-- TITLE -->
  <div class="titleBox">
    <span class="titleSpan"> DONE! </span>
  </div>

  <!-- BODY -->
  <div class="bodyBox">
    <p>
      Experiment completed!  Well done and thanks<br>
      <br>
        Your verification code is
        <a style="background-color:#666; color:#FFF;">
        <?php echo $_POST["participant_id"] ?>
        </a>
        <br>
        
        Copy this and paste into your HIT or you won't get paid!<br>

      <br>
      Remember that if you have any problem you can write back to hiplab@psy.ox.ac.uk.
    </p>
  </div>

  <!-- BUTTON -->
  <button class="buttonBox"
          type="button"
          onClick="offFullscreen();">
         Remove fullscreen
  </button>

</div>
