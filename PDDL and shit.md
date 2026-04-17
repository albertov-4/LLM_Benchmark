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

La prima cosa da capire è che tutte le cartelle contenenti la definzione di un problema in PDDL (e quindi un problema di planning) dovrebbero sempre avere una struttura simile: un file chiamato domain che contiene le cosiddette "regole del gioco" e una cartella che contiene i test set chiamati tutti con il nome "pfile1", "pfile2", ecc.

## File domain
Il file domain contiene la definzione del mondo e le regole che bisogan rispettare; ci si aspetta di trovare le seguenti informazioni:
- Una breve descrizione del problema di planning implementato (tipicamente in inglese)
- La definizione del dominio in PDDL (dal POV logico è assimilabile alla definizione delle variabili e delle classi usate), in particolare si avranno tre principali sezioni:
  - **:types** che è usato per definire la gerarchia degli oggetti nel mondo, nel nostro caso si avrà un qualcosa del tipo "   (:types farm - object)" per indicare che nel mondo esiste una categoria di oggetti chiamata farm. Il trattino (-) indica     l'appartenenza a un tipo. Si possono trovare anche diversi tipi di assegnazioni come ad esempio "(:types plant - thing)".
  - **:predicates** che non sono altro che le proprietà del mondo dati gli oggetti definiti in precedenza, tipicamente se è    stato definito un oggetto in precendenza qui troveremo le relazioni che possono legare diverse istanze di tale oggetto. Ad   esempio nel caso del problema "farmland" si ha "(:predicates (adj ?f1 ?f2 - farm))" che sta a significare che vi è una       relazione di adiacenza "adj" tra due oggetti di tipo farm che qui vengono indicati da dei segnaposto "?f1", "?f2" e il       punto interrogativo sta a significare che quella è una variabile.
  - **:functions** sono funzioni che tipicamente gestiscono numeri reali, possono essere usati per incrementare contatori      numerici (come quello che può rappresentare il costo) oppure per associare un numero a un oggetto del nostro mondo. Nel      caso del nostro problema abbiamo "(:functions (x ?b - farm) (cost))" in cui si defniscono rispettivamente due cose           denotate da parentesi tonde: la prima è un'associazione tra un numero "x" e una variabile denotata con segnaposto "?b" di    tipo farm; la seconda è solo la definzione di un'altra funzione chiamata "cost".
- La definizione delle possibili azioni che sono disponibili nel dominio creato, i requisiti per compiere tali azioni e gli  effetti provocati da tali azioni; ci si aspetta di trovare una struttura in cui prima di tutto si definisce l'azione e il suo nome, dopodichè si devono inserire tre diverse caratteristiche:
  - **:parameters** in cui vengono indicati gli oggetti coinvolti dall'azione, nel nostro problema troviamo la seguente        sintassi ":parameters (?f1 ?f2 - farm)" a indicare che in questo caso gli oggetti coinvolti dall'azione sarano due oggetti   di tipo farm.
  - **:precondition** in cui vengono indicati quali sono i prerequisiti che devono essere presenti per poter effettuare la     tale azione; qui si possono trovare espressioni come "and" e "not", da notare che espressioni con l'and vanno inserite       prima delle quantità su cui devono avere effetto  , quantità che ci si aspetta di trovare racchiuse tra parentesi tonde.     Nel nostro problema si trova la seguente sintassi: ":precondition (and (not (= ?f1 ?f2)) (>= (x ?f1) 4) (adj ?f1 ?f2) )".    Se si procede ad analizzare la sintassi si nota che per riuscire a compiere l'azione associata a questo requisito devono     sussistere contemporaneamente tre condizioni: la prima ci dice dobbiamo verificare la fattoria di partenza e di arrivo       siano diverse e lo si fa con un not; la seconda che la risorsa associata al valore "x" nella fattoria di partenza sia        almeno pari a 4; la terza che ci sia una effettiva adiacenza tra la fattoria di partenza e quella di arrivo.
  - **:effect** non sono altro che le conseguenze che vengono implementate una volta effettuata l'azione, qnche qui possono    sussistere conseguenze multiple se è presente un "and" e nel nostro caso troviamo:":effect (and(decrease (x ?f1) 4)          (increase (x ?f2) 2) (increase (cost) 1))". Si nota che se viene eseguita l'azione si hanno 3 effetti e cioè: il pirmo è     quello di decrementare il valore (e pertanto la risorsa associata) "x" nella fattoria di partenza di un valore pari a 4,     la seconda è quella di incrementare la risorsa e quindi il valore "x" nella fattoria di arrivo di un valore pari a 2 e la    terza è quella di incrementare la funzione costo di un valore pari a 1.

Questo è quello che si trova all'interno del file domani di un problema posto correttamente, di seguito un'immagine completa del file domain riguardante il problema farmland viene allegata:

<img width="551" height="536" alt="Immagine 2026-04-16 112618" src="https://github.com/user-attachments/assets/3826e969-dddf-41fd-a024-8397fd3eabab" />

**Nota:** l'analisi appena fatta riguarda l'azione chiamata "move-fast", si noti anche che negli effetti di questa azione si ha una sorta di perdita di risorse in quanto dalla fattoria di partenza partono 4 di una risorsa indicata con "x" (in questo problema sono workers) e alla fattoria di arrivo arrivano solo 2 workers (l'incremento di x sulla fattoria di arrivo è pari a 2).

## Cartella Instances
Questa cartella contiene tutti i file problem in PDDL, tipicamente sono nominati con il nome "pfileX" dove X è tipicamente un numero da 1 a 20 che sta ad indicare la difficoltà del test; per esempio nel nostro caso il file nominato "pfile1.pddl" contiene solo due oggetti farm e quindi si ha un problema di panning decisamente tendente al caso piu semplice, per avere un parametro di confronto il file "pfile20.pddl" contiene 10 farms che  complicano notevolmente il problema di planning.

**Struttura problem file (pfile)**

Per essere correttamente impostato il problema deve avere la seguente struttura (notare che prima di impostare il problema occorre definire il dominio e quindi redarre il file domain):
- Prima di tutto si definsce il dominio a cui il probelma è associato
- Gli oggetti coinvolti dal problema (nel nostro caso quante farm sono coinvolte): "(:objects farm0 farm1  - farm)"
- Lo stato iniziale da cui bisogan partire a risolvere il problema, nel nostro casoi con 2 sole farm si parte a definire quanti workers ci sono in ogni farm (la risorsa x) e i vincoli che legano le farms, si definisce poi anche il valore delle funzione create nel file domain (nel nostro caso si inizializza il valore della cost function a 0).
- Lo stato finale che si vuole raggiungere (il goal): in questo caso vengono definite le condizioni che devono sussistere per poter ritenere il goal raggiunto e cioè nel nostro esempio e comnde desctritto anche nel file domain l'obiettivo è quello di avere ogni farm con almeno un worker (la riisorsa x). Solitamente si va a imporre anche un vincolo su qualche altra funzione relativa a qualcosa di importante che si vuole tenere d'occhio come il costo in questo caso e in particolare la condizione da rispettare è: "(>= (+ (* 1.0 (x farm0))(+ (* 1.7 (x farm1)) 0)) 840.0)". Per analizzare questa espressione bisogna tenere conto di alcuni concetti importanti tra i quali il fatto che il PDDL lavora con notazione prefissa e ciò vuol dire che gli operatori matematici vengono prima delle quantità su cui agiscono e si deve fare molta attenzione all'ordine delle parentesi per non confondersi. Nella formula sopra riportata possiamo quindi arrivare a dire che la condizione si puo tradurre in " 1* (valore x associato a farm0)+ 1.7*(valore x associato a farm1) + 0 >= 840 "; di fatto è un vincolo per verificare l'utilizzo dei workers sia stato fatto seguendo una certa linea quindi la rete LLM utilizzata deve essere in grado anche di usare le azioni a disposizione per raggiungere questo obiettivo, altra cosa da tenere in considerazione riguardo al problema specifico è che in questo caso si vede come i workers nella farm1 valgano di piu di quelli nella farm0 (si deduce dal valore 1.7 in cui moltiplichiamo x della farm1 nella formula).

Il problema "pfile1.pddl" nella sua interezza è mostrato nella seguente immagine:

<img width="515" height="458" alt="pfile1" src="https://github.com/user-attachments/assets/69de151d-406c-4454-b279-d5e2963e038f" />


## Cenni generali PDDL
In questa sezione verranno accennati diversi concetti fondamentali per la comprensione sia logica che sintattica del PDDL, verranno pertanto presentati sia concetti visti nei due punti precedenti sia concetti totalmente nuovi ma comunque importanti: 
- Un problema espresso in PDDL si divide in due parti distinte: il Dominio (la descrizione del mondo in cui vive il problema) e il Problema di Planning stesso (il problema singolo che va risolto).
- Il PDDL è un liguaggio S-expression (notazione prefissa) il che richiede il fatto che le parentesi aperte vengano sempre chiuse e nel corretto ordine e anche che l'operatore (indipendentemente dal fatto che sia logico o matematico) deve essere inserito sempre prima delle quantita su cui agisce, le quali dovranno essere racchiuse da parentesi anch'esse.
- Il PDDL si basa su assunzione del mondo chiuso (CWA) ciò vuol dire che per quanto riguarda i singoli problemi se ci si dimentica di esplicitare qualcosa nello stato iniziale verrà considerato falso.
- Una funzione importante del PDDL è anche quella di poter definire azioni con una certa durata, ovviamente visto che si parla di azioni queste devono essere definite nel file Domain con la sintassi appropriata, ecco un esempio:
  <img width="530" height="358" alt="time action" src="https://github.com/user-attachments/assets/8dbe2d41-7504-44f6-ba98-fcb7423253ae" />

  Esistono vari vincoli temporali come "(at start (<condition/effect>))" che significa che quello che è specificato deve      accadere all'inizio dell'azione, con sintassi analoga si ha anche "at end"  e poi si può trovare anche "(over all           (<condition>))" che specifica che la condition deve essere vera per tutta la durata dell'azione.
- Sono già state esplorate le funzioni che sono costrutti utili alla gestione delle quantità numeriche che possono variare nel tempo man mano che vengono eseguite azioni; sono costrutti molto importanti in quanto fondamentali per gestire e conoscere i cambiamenti nelle risorse di cui si vuole tenere traccia nel problema: cost function, energy consumption ecc.
- Ci sono poi i processi che sono utili alla rappresentazione di attività continuative nel tempo e che possono continuare a sussistere solo se le condizioni necessarie rimangono valide, se tali condizioni sono verificate i processi sono sempre attivi, ecco un esempio:
  
  <img width="488" height="90" alt="processo" src="https://github.com/user-attachments/assets/2343f1c5-28d1-45e9-8ca8-b3c85b586f52" />

  Come si nota dal codice vengono usati per modificare variabili e costanti numeriche nel tempo man mano che rimangono attivi. Dal punto di vista del coding tradizionale i processi possono essere visti come cicli while.

- Come ultimo costrutto importante ci sono gli eventi che rppresentano occorennze istantanee che possono cambiare alcune proprietà del mondo, una volta che la condizione relativa a un determinato evento diventa vera tale evento accade subito e i suoi effetti si manifestano immediatamente. Ecco un esempio:

  <img width="301" height="94" alt="Evento" src="https://github.com/user-attachments/assets/e8443219-8b25-4a50-b287-7d7cf8a2d37a" />

  Dal punto di vista della programmazone classica il costrutto evento può essere visto come un costrutto IF.



  


  





  





















