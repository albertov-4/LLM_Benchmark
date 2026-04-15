# Tetris Task Family

Questa cartella e lo spazio di lavoro per costruire da zero il dominio `tetris`
o una sua variante per il benchmark.

Struttura:
- `domain/domain.pddl`: definizione del dominio
- `easy/`: istanze semplici
- `medium/`: istanze intermedie
- `hard/`: istanze difficili

Criterio suggerito per la difficolta:
- `easy`: configurazioni con pochi spostamenti critici
- `medium`: piu interazioni geometriche e maggiore profondita del piano
- `hard`: forte dipendenza spaziale e maggiore rischio di invalidita

Convenzione di naming consigliata:
- `instance-01.pddl`
- `instance-02.pddl`
- ...

Fonte di verita:
- il runner deve leggere `domain/domain.pddl`
- il runner deve scoprire le istanze dentro `easy`, `medium`, `hard`
