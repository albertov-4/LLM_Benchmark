# CityCar Task Family

Questa famiglia contiene un dominio starter reale per il benchmark.

Semantica:
- un'auto deve muoversi tra `junction`
- le strade sono rappresentate da archi diretti `road`
- l'unica azione disponibile e `move`

Obiettivo del benchmark:
- verificare se il modello riesce a leggere il grafo delle strade
- mantenere uno stato coerente della posizione dell'auto
- produrre un percorso valido fino alla junction di goal

Struttura:
- `domain/domain.pddl`: dominio `citycar`
- `easy/instance-01.pddl`: percorso corto e quasi lineare
- `medium/instance-01.pddl`: piccolo branching con deviazione inutile
- `hard/instance-01.pddl`: percorso piu lungo con nodi di disturbo e dipendenza a lungo raggio

Lettura della difficolta:
- `easy`: 2 mosse utili
- `medium`: 3 mosse utili con branch ingannevole
- `hard`: 5 mosse utili con piu strade irrilevanti
