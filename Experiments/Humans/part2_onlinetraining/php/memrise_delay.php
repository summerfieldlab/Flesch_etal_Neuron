
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
      Please remember to log back in after three days for the short HIT (5-10 minutes maximum) that completes this task.<br>
      The second task will be available on Thursday from 6pm GMT (1pm EST) and you must complete the task within 24h of that time to receive payment. <br>
      You will receive $8 for completing this second task so don't forget! <br>
      If you have given us your email, we will send you a reminder about this. Thank you for your participation!<br>
      <br>
      <br>
      Remember that if you have any problem you can write back to us at hiplab@psy.ox.ac.uk.
    </p>
  </div>

  <!-- BUTTON -->
  <button class="buttonBox"
          type="button"
          onClick="offFullscreen();">
         Remove fullscreen
  </button>

</div>
