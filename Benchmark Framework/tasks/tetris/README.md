# Tetris Task Family

Questa famiglia contiene un dominio starter reale di riconfigurazione spaziale.

Semantica:
- ogni `piece` occupa una `cell`
- una cella libera e rappresentata da `free`
- l'azione `slide` sposta un pezzo in una cella adiacente libera

Obiettivo del benchmark:
- verificare se il modello tiene traccia dell'occupazione delle celle
- capire se riesce a pianificare mosse in sequenza senza collisioni
- misurare quanto bene gestisce dipendenze spaziali semplici ma cumulative

Struttura:
- `domain/domain.pddl`: dominio `tetris`
- `easy/instance-01.pddl`: un solo pezzo da spostare lungo una catena
- `medium/instance-01.pddl`: due pezzi con dipendenza di ordine
- `hard/instance-01.pddl`: tre pezzi e riconfigurazione piu lunga

Lettura della difficolta:
- `easy`: percorso lineare di 2 mosse
- `medium`: serve liberare e rioccupare celle in ordine corretto
- `hard`: richiede una sequenza piu lunga con piu pezzi che si ostacolano a vicenda
