
/* **************************************************************************************

Shows instructions via html injection
(c) Timo Flesch, 2016 [timo.flesch@gmail.com]
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */

var startedinstructions  = false;
var finishedinstructions = false;
var pageIDX      = 0;
var inStruct     = [];
inStruct.cap     = []; // caption/headline
inStruct.txt     = []; // content
inStruct.img     = []; // illustration

//setInstructions();

function setInstructions(taskID) {
/*
	here I define my instructions
*/
	pageIDX      = 0;
	if(taskID=='dissimrating_pre') {

		inStruct.cap     = []; // caption/headline
		inStruct.txt     = []; // content
		inStruct.img     = []; // illustration
		// welcome (suboptimal, as I define this also in the html)
		inStruct.txt[0]     = "<br>Ahora recibirás instrucciones detalladas para la siguiente tarea. <br> Utiliza los botones de abajo para navegar por estas instrucciones";
		inStruct.img[0]     = [];

		// basics
		inStruct.txt[1]     = "En esta parte del experimento, tendrás que arreglar los árboles en la pantalla de modo que las distancias entre ellos reflejen lo diferentes que parecen uno del otro. <br> <br> Al comienzo de cada ensayo, verás una arena gris con 25 árboles que están ordenados en un círculo. <br> ";
	    inStruct.img[1]     =  "instructions/canvas.png";

		// drag and drop
	    inStruct.txt[2]     = "Si haces clic con el botón izquierdo del ratón sobre un árbol y mantienes presionado el botón, podrás mover el árbol dentro del área gris. <br> <br> Suelta el botón para confirmar su selección. <br> < La animación a continuación ilustra cómo se ve esto en la práctica.";
	    inStruct.img[2]     = "instructions/dragndrop.gif";

	    // dissim ratings
	    inStruct.txt[3]     = "Si crees que dos árboles son muy similares entre sí, muévelos muy cerca el uno del otro. Si parecen diferentes, asegúrate de que estén alejados el uno del otro. <br> Tendrás que hacer esto por <b > TODOS </b> 25 árboles antes de continuar. Es decir, en tu orden final, la distancia entre cualquier par de árboles debe reflejar la diferencia subjetiva entre ellos. Para aclarar, por favor, consulta el siguiente ejemplo.";
	    inStruct.img[3]     = "instructions/arrange.gif";

	    // next
	    inStruct.txt[4]    = "Una vez que estés satisfecho con el orden, presiona el botón verde <b> Próximo ensayo </b> en la parte inferior de la página. <br> Así continuarás con el próximo ensayo, donde se te pedirá que realices la misma tarea con un conjunto de árboles ligeramente diferentes. <br> Ten en cuenta que el orden en ese ejemplo es arbitrario y no contiene ninguna información útil para completar la tarea :) ";
	    inStruct.img[4]    = "instructions/proceed.gif";

	    // summary
	    inStruct.txt[5]    = "<b> Resumen </b> <br> <br> 1. En cada ensayo, verás 25 árboles que están arreglados en un círculo. <br> <br> 2. Tu tarea es cambiar el orden hasta que las distancias entre los 25 árboles reflejen las diferencias entre ellos. <br> <br> 3. Una vez que estés satisfecho con tu orden, haz clic en el botón de próximo ensayo para continuar con un nuevo conjunto de árboles. <br> <br> <br> <b> Habrá seis ensayos que no deberían durar más de 10-15 minutos en total. <br> ¡Si estás listo para comenzar con el experimento, presiona el botón de inicio rojo! </b> ";

	    inStruct.img[5]    = [];
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
	else
		$('.bodyImg').html("<!-- nothing to see here -->");
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
 		$('#nextButton').text('comienzo');
 		$('#nextButton').off('click');
 		if (instr_id=='dissimrating_pre' || instr_id=='dissimrating_post') {
 			$('#nextButton').attr('onclick',"startDissimRatingExperiment()");
 		}
 		else if (instr_id=='treetask_main') {
 			$('#nextButton').attr('onclick',"startMainExperiment()");
 		}

 		$('.buttonBox#nextButton').css('background-color','red');
 	}
}
