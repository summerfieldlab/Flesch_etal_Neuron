function inStruct = setInstructions_refresher()
	%% SETINSTRUCTIONS_PRE_BEHAV()
	%
	% sets instructions as struct
	% with fields for text and image
	%
	% Timo Flesch, 2018

	inStruct = struct();

	% summary
	inStruct(1).txt   = 'RESUMEN   \n 1. Hay dos huertos.  \n \n 2. 2. En cada ensayo, un cuadrado de color te dirá en qué huerto estás actualmente . \n \n 3. Cuadrado azul: huerto norte. Cuadrado naranjo: huerto sur \n\n 4. Esto puede o no cambiar entre los ensayos. ¡Presta atención! \n \n 5. Diferentes tipos de árboles crecen lo mejor en los dos diferentes huertos. \n \n 6. Las reglas son las mismas de ayer – haces esta tarea para que te recuerdes de las reglas. \n \n 7. Usa las teclas F, J (izquierda, derrecha) para comunicar su decisión.  \n \n 8. Preste atención a la asignación que se muestra en la pantalla. \n \n  Eso te indica si la tecla F o J corresponde a ‘aceptar’ \n \n 9. Maximiza tu recompensa! \n \n Pulsa la banda espaciadora para continuar.';
	inStruct(1).img   = '';



end
