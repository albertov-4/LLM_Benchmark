# Classificazione planning problems
Nel nostro caso nella cartella planning domains sono presenti svariati problemi di planning di diversa tipologia e difficoltà sorge quindi la necessità di capire come classificarli prima ancora di fare una scelta di domini papabili per creare il benchmark.
Una soluzione pratica potrebbe essere quella di basare la classificazione prima sul tipo di planning richiesto:

<img width="731" height="438" alt="Immagine 2026-04-15 154014" src="https://github.com/user-attachments/assets/45055bab-0712-4291-b23e-e4f6376f16bc" />

 In modo tale da garantire una sufficiente variabilità e applicabilità del benchmark.

 Dopodichè si potrebbe passare a una classificazione basata sulla complessità del dominio specifico:
 
<img width="763" height="353" alt="Immagine 2026-04-15 155650" src="https://github.com/user-attachments/assets/f91ee794-aa8a-47c3-82f4-ffc4fa8194f1" />

Applicando queste due classificazioni ed ipotizzando di scegliere 3 diversi problemi classificati in base al tipo di planning si dovrebbe poi andare a decidere di classificare in base alla difficoltà e per ogni progetto scelto si prevede un massimo di 3 livelli di difficoltà: facile, medio e difficile. In totale tuttavia in questo modo si avrebbero 3 diversi problemi su cui lavorare 3 istanze diverse ciascuno (quindi in totale 9 problemi).

# Comprensione e sintassi PDDL
La sezione corrente sarà dedicata alla comprensione logica e sintattica del linguaggio PDDL (Planning Domain Definition Language), standard per la definzione di planning problems. Aprendo la cartella planning domains ci troveremo davanti a una serie di problemi, per questo esempio è stato scelto il problema chiamato "farmland"; dentro alla cartella ci si troverà davanti a un file e a un'altra cartella instances (contenente i test set):

<img width="914" height="179" alt="Immagine 2026-04-15 164621" src="https://github.com/user-attachments/assets/01e46bfb-cfc2-4a89-a30e-4c233c95df62" />

La prima cosa da capire è che tutte le cartelle contenenti la definzione di un problema in PDDL (e quindi un problema di planning) dovrebbero sempre avere una struttura simile: un file 
