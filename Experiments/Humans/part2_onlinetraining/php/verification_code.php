
<div class="screenBox">

  <!-- TITLE -->
  <div class="titleBox">
    <span class="titleSpan"> ¡TERMINADO! </span>
  </div>

  <!-- BODY -->
  <div class="bodyBox">
    <p>
      ¡Experimento completado! Bien hecho y gracias.<br>
      <br>
        Tu código de verificación es
        <a style="background-color:#666; color:#FFF;">
        <?php echo $_POST["participant_id"] ?>
        </a>
        <br>

        Copia esto y envíalo al experimentador, ¡o no recibirás el pago!
<br>

      <br>
      Recuerda que si tienes algún problema, puedes escribir a hiplab@psy.ox.ac.uk.
    </p>
  </div>

  <!-- BUTTON -->
  <button class="buttonBox"
          type="button"
          onClick="offFullscreen();">
         salir de pantalla completa
  </button>

</div>
