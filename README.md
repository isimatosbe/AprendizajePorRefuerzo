# Aprendizaje por Refuerzo

Código de las implementaciones del trabajo de fin de máster *Aprendizaje por Refuerzo* de Isidro Matos Bellido.

## Índice
- [Enseñando a un agente a jugar al blackjack](#blackjack)
- [Problema del taxista](#taxista)

## Enseñando a un agente a jugar al blackjack <a name="blackjack"></a>

Archivo `blakcjack.ipynb`.

Nuestro objetivo será enseñar a un agente a jugar al blackjack mediante un método de Monte Carlo para políticas $\varepsilon$-suaves. Usaremos la implementación de OpenAI Gym para generar las partidas de blackjack, esta interfaz muestra los estados como puede observarse en la imagen a continuación.

<p align="center">
  <img src="img/blackjack/blackjack.png" style="width: 35%">
</p>
  
Aplicamos el método de Monte Carlo para políticas $\varepsilon$-suaves, tomando como política inicial la política aleatoria e imponiendo como condición de parada que cada uno de los 560 pares estado-acción sea visitado, al menos, 1000 veces. Tras entorno a 50 minutps de cálculo y unas 45 millones de partidas obtenemos la siguiente política óptima:

<table><tr>
  <td> <p align="center"> <img src="img/blackjack/usable.jpeg" style="width: 75%"> </p> </td>
  <td> <p align="center"> <img src="img/blackjack/noUsable.jpeg" style="width: 75%"> </p> </td>
</tr></table>

Los valores en azul son aquellos en los que la política que hemos obtenido no coincide con la planteada por Richard S. Sutton y Andrew G. Barto en Reinforcement Learning, esta diferencia no es muy significativa pues la función de valor en estos dos pares estado-acción varía en el tercer decimal, ambas acciones (*hit* y *stick*) tienen una recompensa esperada muy similar.

## Problema del taxista <a name="taxista"></a>

Archivo `taxista.ipynb`.

El problema del taxista consiste en entrenar a un agente, que vive en un tablero de 5x5, para que sea capaz de recoger a un cliente y llevarlo a su destino. Al igual que antes usamos la interfaz facilitada por OpenAI Gym para generar los distintos episodios del entrenamiento. Un estado inicial obtenido de esta forma puede observarse a continuación.

<p align="center"> <img src="img/taxi/taxi.png" style="width: 40%"> </p>

 En este caso entrenaremos dos agentes y los compararemos, uno de ellos seguirá el algoritmo Sarsa mientras que el otro seguirá el algoritmo de $Q$-*learning*. El entrenamiento de ambos se hará usando los mismos estados iniciales para asegurarnos de que la comparación es lo más realista posible. Pararemos el entrenamiento de ambos tras un número fijo de episodios, como es un ejemplo sencillo lo fijaremos como 5000, con este número tardaremos unos 10 segundos en entrenar a ambos agentes.

 Tras el entrenamiento se obtiene que ambos agentes obtienen una recompensa media en sus mejores 100 episodios (seguidos) de más de 8.5 en la inmensa mayoría de las ejecuciones, estos resultados pueden considerarse bastante buenos siguiendo la [clasificación](https://github.com/openai/gym/wiki/Leaderboard) facilitada en el repositorio de GitHub de OpenAI Gym.

 La comparativa entre las recompensas medias para ambos algoritmos puede observarse en la imagen a continuación, como puede verse ambos agentes tienen un rendimiento bastante parecido.

 <p align="center"> <img src="img/taxi/comparacion.jpeg" style="width: 100%"> </p>

 Además de esta comparativa, para comprobar realmente el entrenamiento, se han grabado algunas de los episodios por los cuales han pasado los agentes durante el entrenamiento, estos pueden verse a través de la siguiente [lista de reproducción](https://youtube.com/playlist?list=PLzjBjc6HHLwhhE-Pdzvxjl0eR0Ah9mSr8) de YouTube.