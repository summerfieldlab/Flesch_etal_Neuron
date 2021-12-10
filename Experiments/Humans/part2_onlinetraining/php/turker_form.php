<div class="screenBox">

  <!-- TITLE -->
  <div class="titleBox">
    <span class="titleSpan"> PARTICIPANT FORM </span>
  </div>

  <!-- FORM -->
  <form action="javascript:getTurkerform();goWebsite(html_instructions);" method="POST">
    <!-- BODY -->
    <div class="bodyBox">
      <p>We would like to ask you some details about yourself.  We will not record anything about you that can identify you.  This is just so we can see whether there are any trends within the data.</p>
      <p>Also, the MTurker ID will be used to make sure to make sure we pay you!</p>
      <br>
      <table style="position:         absolute;
                    width:            300px;
                    left:             50%;
                    margin-left:      -150px;
                    border-collapse:  collapse;">
        <tr>
          <td style="padding-top:10px; padding-bottom:10px;">
            <b>Gender</b>
          </td>
          <td>
            <select id="genderSelect" required>
                <option selected value=""      >Select</option>
                <option          value="male"  >Male  </option>
                <option          value="female">Female</option>
            </select>
          </td>
        </tr>
        <tr>
          <td style="padding-top:10px; padding-bottom:10px;">
            <b>Age</b>
          </td>
          <td>
            <select id="ageSelect" required>
              <option selected value=""     >Select</option>
              <option          value="18-20">18-20 </option>
              <option          value="21-30">21-30 </option>
              <option          value="31-40">31-40 </option>
              <option          value="41-50">41-50 </option>
              <option          value="51-60">51-60 </option>
              <option          value="61+"  >61+     </option>
            </select>
          </td>
        </tr>
        <tr>
          <td style="padding-top:10px; padding-bottom:10px;">
            <b>Amazon Mechanical Turk Worker ID</b>
          </td>
          <td>
            <input type="text" id="turkerSelect" required title="Insert your Mturk ID with uppercase and without spaces!" value=<?php echo "\"".$_GET["mturkid"]."\"" ?> disabled><br>
          </td>
        </tr>
        <tr>
          <td style="padding-top:10px; padding-bottom:10px;">
            <b>Do you speak Lithuanian?</b>
          </td>
          <td>
            <select id="lithuanianSelect" required>
                <option selected value=""   >Select</option>
                <option          value="yes">Yes</option>
                <option          value="no" >No </option>
            </select>
          </td>
        </tr>
        <tr>
          <td style="padding-top:10px; padding-bottom:10px;">
            <b>Are you a native English speaker?</b>
          </td>
          <td>
            <select id="englishSelect" required>
                <option selected value=""   >Select</option>
                <option          value="yes">Yes</option>
                <option          value="no" >No </option>
            </select>
          </td>
        </tr>
      </table>

    </div>

    <!-- BUTTON -->
    <button class="buttonBox"
            type="submit">
           Submit
    </button>
    
  </form>

</div>
