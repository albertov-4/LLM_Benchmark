# CityCar Task Family

Questa cartella e lo spazio di lavoro per costruire da zero il dominio `citycar`
o una sua variante per il benchmark.

Struttura:
- `domain/domain.pddl`: definizione del dominio
- `easy/`: istanze semplici
- `medium/`: istanze intermedie
- `hard/`: istanze difficili

Criterio suggerito per la difficolta:
- `easy`: pochi oggetti, piani brevi, branching basso
- `medium`: piu vincoli e maggiore bisogno di tenere traccia dello stato
- `hard`: dipendenze a lungo raggio, piani lunghi, maggiore rischio di errori

Convenzione di naming consigliata:
- `instance-01.pddl`
- `instance-02.pddl`
- ...

Fonte di verita:
- il runner deve leggere `domain/domain.pddl`
- il runner deve scoprire le istanze dentro `easy`, `medium`, `hard`
