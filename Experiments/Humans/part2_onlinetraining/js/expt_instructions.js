
/* **************************************************************************************

Shows instructions via html injection

Timo Flesch, 2018,
Human Information Processing Lab,
Experimental Psychology
University of Oxford


************************************************************************************** */

var startedinstructions  = false;
var finishedinstructions = false;
var pageIDX      = 0;
var inStruct     = [];
inStruct.cap     = []; // caption/headline
inStruct.txt     = []; // content
inStruct.img     = []; // illustration

//setInstructions();

function setInstructions() {
/*
	here I define my instructions
*/
	pageIDX      = 0;

	if (coding.phase==1) {
		console.log('phase 1 instructions');
		inStruct.cap     = []; // caption/headline
		inStruct.txt     = []; // content
		inStruct.img     = []; // illustration
		// welcome (suboptimal, as I define this also in the html)
		inStruct.txt[0]     = "<br>  Ahora vas a recibir instrucciones detalladas. <br> Por favor utiliza los botones para navegar las instrucciones. ";
		inStruct.img[0]     = [];


		// whereami
		inStruct.txt[1]     = "Estás en la PRIMERA fase del experimento.";
		inStruct.img[1]     = 'instructions/task_structure_today_p1.png';

		// north orchard
		inStruct.txt[2]     = "Imagínate que estés un jardinero. <br> Posees dos huertos.<br> El primer es <b> el huerto norte </b>, señalado por un rectángulo azul <br>";
	    inStruct.img[2]     =  "instructions/intro_north.png";

	    // south orchard
	    inStruct.txt[3]     = "El segundo huerto es <b>el huerto sur</b>, señalado por un rectángulo naranjo <br>";
	    inStruct.img[3]     =  "instructions/intro_south.png";

		// what to do
		inStruct.txt[4]     = "<b> Te gustaría plantear unos árboles en tus huertos <br><br>Por desgracio ya no sabes cuales árboles pueden crecer en tus huertos.<br>Solo sabes que hay un tipo de árboles que crece lo mejor en el huerto norte<br> y otro tipo que crece lo mejor en el huerto sur, <br><br><b>Tu objetivo estará aprender cual tipo de árbol crece en el huerto sur y cual tipo crece lo mejor en el huerto norte para que maximizes tu recompensa.";
		inStruct.img[4]    = [];

		// trial - context
		inStruct.txt[5]     = "En la fase del aprendizaje cada ensayo empieza con un imagen del huerto en el que estás.";
		inStruct.img[5]     = "instructions/intro_north.png";

		// trial - stimulus
		inStruct.txt[6]     = "Después aparecerá un imagen de un árbol y la asignación de teclas .<br> La asignación de teclas cambia aleatoriamente entre ensayos, y por eso te pedimos que prestes atención!!";
		inStruct.img[6]     =  "instructions/intro_stim.png";

		// trial - decision
		inStruct.txt[7]     = "Decides si quieres plantear el árbol o no. <br> <b>Para comunicar tu decisión por favor pulsa las teclas F or J (corresponden a la izquierda y derecha de las opciones en la pantalla) </b> <br> En este ejemplo, pulsé la tecla F, que está asignada a <i>aceptar</i> en este ensayo <br> La opción que eliges estará estacada con un rectángulo negro:";
		inStruct.img[7]     = "instructions/intro_accept.png";

		// trial - feedback 1
		inStruct.txt[8]     = " Pues verás si tu decisión era corecta o no. <br> Verás la recompensa de ambas opciones <br> Tu selección también estará estacada con un rectángulo negro. <br> En este ejemplo recibiste recompensa, indicada por un número verde:";
		inStruct.img[8]     = "instructions/intro_feedback_pos.png";

		// trial - feedback 2
		inStruct.txt[9]    = "Recompensas (valores positivos) siempre estarán verdes, pero recargos (valores negativos) estarán en rojo"; 
		inStruct.img[9]    = "instructions/intro_feedback_neg.png";

		// trial - feedback 3
		inStruct.txt[10]    = "Si no aceptas un árbol siempre recibirás nada por tu selección. <br> Pero de vez en cuando no aceptar es la opción correcta porque hay árboles que no crecen bien en el huerto en el que estás. <br> En este ejemplo recibiste recargo por haber elegido plantear el árbol. Por eso la respuesta correcta en este caso es no aceptar el árbol y no recibir recompensa (evitando recargos):";
		inStruct.img[10]    = "instructions/intro_feedback_reject_good.png" ;

		// trial - feedback 4
		inStruct.txt[11]    = "Recompensas son entre -50 y +50 <br> Tu tarea es aprender cual tipo de árbol crece lo mejor en cada huerto, <br> y sólo plantear (por ejemplo <i>aceptar</i>) ese tipo. <br> Rechaza los arboles que no crecen bien.";
		inStruct.img[11]    = [];

		// trial - feedback 5
		inStruct.txt[12]    = "Pero trata de ser rápido! Si esperas demasiado tiempo, se rechazará automáticamente el árbol";
		inStruct.img[12]    = "instructions/intro_feedback_reject_bad.png"; //TODO change, this is a placeholder

		// reminder
		inStruct.txt[13]    = "Habrá fases de FORMACIÓN y PRUEBA.  <br>  Verás si tu respuesta es correcta sólo en la fase de formación. <br> La fase de prueba es para evaluar si has aprendido las reglas. <br> Habrá descansos entre las fases. <br> Ahora estás en la fase de formación.";
		inStruct.img[13]    = [];

	    // structure
	    inStruct.txt[14]    ="RESUMEN   <br> 1. Hay dos huertos.  <br> <br> 2. En cada ensayo, un cuadrado de color te dirá en qué huerto estás actualmente. <br> <br> 3. Esto puede o no cambiar entre los ensayos. ¡Presta atención! <br> <br> 4. Diferentes tipos de árboles crecen lo mejor en los dos diferentes huertos. <br> <br> 5. Aprende cual tipo debes plantear en cada huerto y cual tipo no. <br> <br> 6. Usa las teclas F, J (izquierda, derrecha) para comunicar su decisión. <br> <br> 7. Preste atención a la asignación que se muestra en la pantalla. <br> <br>  Eso te indica si la tecla F o J corresponde a  <i>aceptar</i>. <br> <br> 8. Maximiza tu recompense! <br> <br>";
	    inStruct.img[14]    = [];
		// // summary
		inStruct.txt[15]   = "Pulsa el botón <i>inicio</i> abajo para empezar la fase de formación";
		inStruct.img[15]   = [];

	}

	else if (coding.phase==2) {
		console.log('phase 2 instructions');

		inStruct.cap     = []; // caption/headline
		inStruct.txt     = []; // content
		inStruct.img     = []; // illustration
		inStruct.txt[0]     = "<br> Ahora recibirás instrucciones detalladas para la siguiente fase. <br> Utilice los botones de abajo para navegar a través de estas instrucciones.";
		inStruct.img[0]     = [];

		inStruct.txt[1]     = "Ahora estás en la segunda parte del experimento.";
		inStruct.img[1]     = 'instructions/task_structure_today_p2.png';


		inStruct.txt[2]     = "Como siempre, cada ensayo empieza con una imagen del huerto en el que te encuentras ahora, <br> por ejemplo el huerto sur";
		inStruct.img[2]     = "instructions/intro_south.png";

	 	inStruct.txt[3]     = "Después aparecerá un imagen de un árbol y la asignación de teclas .<br> La asignación de teclas cambia aleatoriamente entre ensayos, y por eso te pedimos que prestes atención!!";
	 	inStruct.img[3]     = "instructions/intro_stim.png";

		inStruct.txt[4]     = "Decides si quieres plantear el árbol o no. <br> <b>Para comunicar tu decisión por favor pulsa las teclas F or J (corresponden a la izquierda y derecha de las opciones en la pantalla) </b> <br> En este ejemplo, pulsé la tecla F, que está asignada a <i>aceptar</i> en este ensayo <br> La opción que eliges estará estacada con un rectángulo negro:";
	 	inStruct.img[4]     = "instructions/intro_accept.png";

		inStruct.txt[5]     = "Hasta aquí, todo es exactamente lo mismo como en las fase de formación. <br> Pero, como nos interesa si has aprendido las reglas durante la formación, <br> no verás si tu respuesta es corecta o incorecta en esa fase";
	 	inStruct.img[5]     = [];

		inStruct.txt[6]     = "Eso signidica que después de comunicar tu decisión  (aceptar o rechazar) <br> no verás si tu decisión era corecta, sino habrá una pantalla gris antes del próximo ensayo.";
	 	inStruct.img[6]     = [];

		inStruct.txt[7]     = "RESUMEN   <br> 1. Hay dos huertos.  <br> <br> 2. En cada ensayo, un cuadrado de color te dirá en qué huerto estás actualmente. <br> <br> 3. Esto puede o no cambiar entre los ensayos. ¡Presta atención! <br> <br> 4. Diferentes tipos de árboles crecen lo mejor en los dos diferentes huertos. <br> <br> 5. Aprende cual tipo debes plantear en cada huerto y cual tipo no. <br> <br> 6. Usa las teclas F, J (izquierda, derrecha) para comunicar su decisión. <br> <br> 7. Preste atención a la asignación que se muestra en la pantalla. <br> <br>  Eso te indica si la tecla F o J corresponde a  <i>aceptar</i>. <br> <br> 8. Maximiza tu recompense! <br> <br>";
	 	inStruct.img[7]     = [];

		// // summary
		inStruct.txt[8]   = "Pulsa el botón <i>inicio</i> abajo para empezar la fase de formación";
		inStruct.img[8]   = [];
	}
	else if (coding.phase==3) {
		console.log('phase 3 instructions');

		inStruct.cap     = []; // caption/headline
		inStruct.txt     = []; // content
		inStruct.img     = []; // illustration
		inStruct.txt[0]     = "<br> Recibirás instrucciones detalladas para la siguiente fase. <br> Utilice los botones de abajo para navegar a través de estas instrucciones.";
		inStruct.img[0]     = [];
		// sunnary
		inStruct.txt[1]     = "<b> Ahora estás en la TERCERA fase. De nuevo habrá formación. Las reglas son las mismas. Queremos asegurarnos de que recibas suficiente formación para la sesión de mañana. Esta es la fase más larga, debería tomar alrededor de 60 minutos.</b>";
		inStruct.img[1]     = 'instructions/task_structure_today_p3.png';

		inStruct.txt[2]     = "RESUMEN   <br> 1. Hay dos huertos.  <br> <br> 2. En cada ensayo, un cuadrado de color te dirá en qué huerto estás actualmente. <br> <br> 3. Esto puede o no cambiar entre los ensayos. ¡Presta atención! <br> <br> 4. Diferentes tipos de árboles crecen lo mejor en los dos diferentes huertos. <br> <br> 5. Aprende cual tipo debes plantear en cada huerto y cual tipo no. <br> <br> 6. Usa las teclas F, J (izquierda, derrecha) para comunicar su decisión. <br> <br> 7. Preste atención a la asignación que se muestra en la pantalla. <br> <br>  Eso te indica si la tecla F o J corresponde a  <i>aceptar</i>. <br> <br> 8. Maximiza tu recompense! <br> <br>";
		inStruct.img[2]     = "";
		// // summary
		inStruct.txt[3]   = "Pulsa el botón <i>inicio</i> abajo para empezar la fase de formación";
		inStruct.img[3]   = [];


	}


}




function gotoNextPage() {
/*
	changes div to next entry in instruction array
*/
	// move forward
	pageIDX++;
	changeInstructions();
	changeButtons();
}

function gotoPrevPage() {
/*
	changes div to previous entry in instruction array
*/
	// move backward
	pageIDX--;
	changeInstructions();
	changeButtons();
}


function changeInstructions() {
/*
	changes div content via html injection
*/

	$('.bodyText').html(inStruct.txt[pageIDX]);
	if (inStruct.img[pageIDX].length>0) {
		$('.bodyImg').html("<img id=instr_img src=" + inStruct.img[pageIDX] + ">");
	}
	else{
		$('.bodyImg').html("<!-- nothing to see here -->");
	}
}


function changeButtons() {
/*
	changes properties of buttons
*/
	console.log(pageIDX);

	if (pageIDX == 0) {
		$('#prevButton').prop('disabled', true);
	}
	else {
		$('#prevButton').prop('disabled', false);
	}

	if (pageIDX == inStruct.txt.length-2) {
 		$('#nextButton').text('Siguiente página');
 		$('#nextButton').off('click');
 		$('#nextButton').attr('onclick',"gotoNextPage()");
 		$('.buttonBox#nextButton').css('background-color','rgba(249,167,50,1)');
 	}
	if (pageIDX == inStruct.txt.length-1) {
 		$('#nextButton').text('Inicio');
 		$('#nextButton').off('click');
		$('#nextButton').attr('onclick',"startPhase()");
		$('.buttonBox#nextButton').css('background-color','red');
 		}



}
