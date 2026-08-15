# tiny-vllm — Corso (traduzione italiana)

> **Nota di Claude**: questa è una traduzione italiana del README/corso originale di [Jędrzej Maczan](https://github.com/jmaczan/tiny-vllm) (repo `jmaczan/tiny-vllm`, licenza Apache-2.0). Ho tradotto la prosa mantenendo il codice come nell'originale (variabili, commenti e nomi di funzione restano in inglese, come è normale per il codice sorgente). Le sezioni che nell'originale erano segnate come "Incoming!" (non ancora scritte dall'autore) le ho completate io con contenuto originale — **queste parti sono chiaramente marcate più sotto** con un riquadro di avviso, così sai sempre distinguere il testo di Maczan dalla mia interpretazione. Quando l'autore pubblicherà la versione ufficiale di quelle sezioni, ti consiglio di confrontarla con quanto scritto qui.

Stai per costruire un motore di inferenza LLM ad alte prestazioni con C++ e CUDA — tiny-vllm, un fratello minore e più piccolo di [vLLM](https://github.com/vllm-project/vllm).

Impareremo molto lungo il percorso, faremo errori, e deriveremo idee e matematica da zero.

Questo repository è composto da due cose: 1. il codice sorgente completo del server di inferenza e 2. un corso in cui ti guido attraverso il processo di implementazione del motore. Sentiti libero di usarlo come strumento di apprendimento nel tuo percorso, oppure, se sei un docente, come risorsa didattica per la tua università.

Il motore di inferenza consiste in:

- [x] caricare un vero modello LLM da Safetensors (Llama 3.2 1B Instruct)
- [x] forward pass LLM completo (prefill + decode)
- [x] tutto il calcolo con kernel CUDA
- [x] KV cache
- [x] batching statico
- [x] batching continuo
- [x] [online softmax, stile FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [x] [PagedAttention](https://arxiv.org/pdf/2309.06180)

Preparati una bevanda calda e cominciamo.

## Indice

- [Introduzione: LLM, vLLM, modelli, server di inferenza](#introduzione-llm-vllm-modelli-server-di-inferenza)
- [Prerequisiti tecnici](#prerequisiti-tecnici)
- [Safetensors e il tuo modello](#safetensors-e-il-tuo-modello)
- [Come funzionano i numeri floating-point e perché usiamo bfloat16](#come-funzionano-i-numeri-floating-point-e-perché-usiamo-bfloat16)
- [Memoria GPU e CPU](#memoria-gpu-e-cpu)
- [Inferenza di un singolo token](#inferenza-di-un-singolo-token)
- [Tokenizzazione](#tokenizzazione)
- [Embedding](#embedding)
- [Ingegneria di kernel CUDA — embedding](#ingegneria-di-kernel-cuda--embedding)
- [RMSNorm e riduzione parallela in CUDA](#rmsnorm-e-riduzione-parallela-in-cuda)
- [RoPE](#rope)
- [Connessioni residue](#connessioni-residue)
- [cublasGemmEx](#cublasgemmex)
- [Il trucco di trasposizione column-major → row-major](#il-trucco-di-trasposizione-column-major--row-major)
- [Prefill vs decode](#prefill-vs-decode)
- [Perché esiste il KV cache](#perché-esiste-il-kv-cache)
- [Attention](#attention)
- [GQA](#gqa)
- [SiLU](#silu)
- [Softmax](#softmax)
- [Maschera causale](#maschera-causale)
- [Argmax](#argmax)
- [Feed forward network](#feed-forward-network)
- [Riuso dei buffer](#riuso-dei-buffer)
- [Batching statico](#batching-statico)
- [Batching continuo](#batching-continuo)
- [Online softmax](#online-softmax)
- [PagedAttention](#pagedattention) ⚠️ *sezione integrata da Claude*
- [Paged KV cache](#paged-kv-cache) ⚠️ *sezione integrata da Claude*
- [Kernel CUDA per Paged Attention](#kernel-cuda-per-paged-attention) ⚠️ *sezione integrata da Claude*

---

## Introduzione: LLM, vLLM, modelli, server di inferenza

È facile perdersi con tutto quello che succede negli ultimi anni. Cerchiamo di fare chiarezza.

**Un LLM è un modello**. Fisicamente, **un LLM è un file che contiene un sacco di [numeri floating-point](https://en.wikipedia.org/wiki/Floating-point_arithmetic)**. Concettualmente, questi numeri rappresentano i pesi di alcune operazioni. I pesi vengono appresi/scoperti/trovati durante la fase di training. Alcune operazioni usano questi pesi. Ogni operazione è una funzione, che prende dei dati in input, ci fa qualcosa e produce dati in output. Le operazioni e il loro ordine sono definiti dall'architettura dell'LLM. Ogni modello ha la propria architettura, progettata da ingegneri e ricercatori.

Il processo che va da 0 a un LLM che scrive testo è così:

0. **Progettare il modello** — ingegneri e ricercatori usano un linguaggio ad alto livello come Python con una libreria per tensori come [PyTorch](https://github.com/pytorch/pytorch) o [tinygrad](https://github.com/tinygrad/tinygrad) per progettare l'architettura del modello. Addestrano versioni piccole del modello, fanno esperimenti con operazioni, dati e iperparametri (parametri delle operazioni) diversi. È la fase in cui si definisce la specifica.
1. **Implementare il modello** — Una volta decisa l'architettura finale e preparati i dati per il training, scrivono il codice che definisce il modello finale. Può essere ancora in PyTorch o simili.
2. **Addestrare il modello** — L'architettura scelta viene inizializzata con pesi fittizi. Scrivono uno script che usa ancora PyTorch o simili per far girare un algoritmo di apprendimento come la backpropagation su molto hardware, come [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) e [TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit). Questa fase brucia molta energia, denaro e potenza di calcolo. Il prodotto della fase di training è un file con i pesi del modello, in qualche formato, come il [formato Safetensors](https://huggingface.co/docs/safetensors/index). Quindi, la fase di training consiste nel trovare un insieme di pesi che produca un buon testo usando l'architettura data.
3. **Servire il modello (siamo qui)** — Il file con i pesi non può essere eseguito su un computer. Non è un eseguibile. È tanti numeri. Neanche l'architettura può essere eseguita — è solo un piano, un progetto, una descrizione del calcolo. Per far girare davvero il modello, serve un programma che trasformi l'architettura e le sue operazioni in codice eseguibile e che usi il file dei pesi per caricarli nell'architettura. Una volta scritto un programma che implementa le operazioni e che carica i pesi (i pesi vengono caricati a runtime, all'avvio del programma), puoi finalmente inviare prompt al modello e ottenere una risposta sensata. Generare un output da un modello si chiama inferenza. Ecco perché quello che costruiamo qui si chiama server di inferenza o motore di inferenza.

Conoscendo il motivo per cui serve un server di inferenza, vediamo perché lo costruiamo in C++ e CUDA. È perché vogliamo massimizzare l'uso efficiente dell'hardware e ottenere alte prestazioni. Significa che vogliamo risposte veloci ed essere in grado di gestire più prompt contemporaneamente. CUDA è l'intero ecosistema, ma anche un linguaggio che usi per scrivere codice che gira sulle GPU. Dobbiamo scrivere codice per GPU perché molte operazioni dentro un LLM consistono nel moltiplicare e sommare molti numeri. Se devi fare poca matematica, la CPU basta. Se ne devi fare tanta, la GPU è meglio. Gli LLM riguardano soprattutto la moltiplicazione di matrici, che si riduce al calcolo di prodotti scalari di due vettori, per molti numeri e per molti vettori. La matematica degli LLM è semplice, ci serviranno le basi di algebra lineare e puoi impararle mentre scrivi codice, colmando le lacune strada facendo. Trovo che questo modo di imparare "just-in-time" sia il più efficace, e forse piacerà anche a te.

La mia visione sul rapporto tra IA e calcolo, che magari trovi utile, è che **l'intelligenza deriva da tanti parametri del modello e tanto calcolo dei valori in input usando questi parametri**. Non c'è un singolo elemento a cui puoi puntare e dire: "questo è ciò che rende il modello intelligente o utile". Ogni parte del modello puoi sostituirla con una diversa e ottenere compromessi diversi, come scambiare accuratezza per complessità. Spero di non dimenticarmi di tornare su questo argomento più avanti, quando toccheremo la matematica dell'attention. Perché — il meccanismo di attention di default è molto costoso computazionalmente (O(n²·d)). E questa complessità può essere messa in discussione, e infatti alcuni lo fanno, trovando meccanismi di attention alternativi, come l'[attention lineare](https://haileyschoelkopf.github.io/blog/2024/linear-attn/). Se questo corso risulterà utile a più persone, penserò a farne un altro, sui compilatori ML (uno pratico in Python o C++ più un po' di teoria SSA) o sui meccanismi di attention alternativi (matematica + kernel CUDA). Se sei interessato, fammelo sapere! Se questo corso ti risulterà utile, per favore fallo sapere ad altre persone.

> Fuori campo: la fase di training di un LLM è qualcosa che non facciamo in questo corso. Prendiamo un LLM già addestrato e scriviamo un programma che lo farà girare velocemente su GPU NVIDIA per più richieste in parallelo. Se vuoi addestrare il tuo LLM, ti consiglio vivamente i repository del maestro Karpathy come [nanoGPT](https://github.com/karpathy/nanoGPT) e [llm.c](https://github.com/karpathy/llm.c) e il suo [canale YouTube](https://www.youtube.com/@AndrejKarpathy). Allo stesso modo, non progettiamo il modello, ma anche le librerie per tensori sono un argomento affascinante e vale la pena capirle da zero. [tinygrad](https://github.com/tinygrad/tinygrad) di George Hotz è un progetto che implementa una libreria per tensori con pochissimo codice, quindi se vuoi ispirarti e capire gli interni, è un buon posto (anche il loro Discord è carino). C'è anche una versione più vecchia e piccola di Andrej Karpathy — [micrograd](https://github.com/karpathy/micrograd). E visto che ho citato Discord, ti consiglio il [GPU MODE](https://discord.com/invite/gpumode) di [Mark Saroufim](https://www.marksaroufim.com/). Un sacco di persone in gamba ci girano! E se ti senti perso con quello che succede qui, e sei nuovo nel tuo percorso AI/ML, inizia con il [libro fastai](https://course.fast.ai/Resources/book.html) di Jeremy Howard e Rachel Thomas. Ometto volutamente la parte di data science ed engineering qui, perché non ne so molto. Probabilmente [Kaggle](https://www.kaggle.com/) può essere un buon punto di partenza per imparare facendo. Ultimo ma non meno importante, programmeremo in C++ e CUDA e useremo [cuBLAS](https://developer.nvidia.com/cublas) dove applicabile. Puoi imparare strada facendo. Le [risorse ufficiali NVIDIA](https://docs.nvidia.com/cuda/cuda-programming-guide/) sono buone e utili.

## Prerequisiti tecnici

Puoi costruirlo ed eseguirlo su qualsiasi piattaforma, con piccole modifiche, ammesso che tu abbia una GPU NVIDIA. Potresti dover aggiustare alcuni percorsi, come CUDA o GCC in `c_cpp_properties.json` o NVCC in `CMakeLists.txt`.

Ti suggerisco di fare un fork di questo repo e fare gli aggiustamenti necessari perché funzioni sulla tua macchina, poi creare una pull request verso `jmaczan/tiny-vllm` per condividere a monte le tue modifiche a beneficio di altri lettori.

Il setup esatto su cui l'autore sviluppa e testa:

- Linux (6.19.8 x64_64)
- CUDA Toolkit (13.1)
- C++ 17
- GCC (15.2.1)
- L'unica dipendenza esterna che scaricherai è il parser JSON [nlohmann/json](https://github.com/nlohmann/json) 3.12.0, che è un singolo file header (`include/json.hpp`)
- CPU AMD (Ryzen 7 9800X3D)
- GPU NVIDIA (RTX 5090)
- Usato [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) da Hugging Face (commit hash `898999bd25b40516fce5a5b8f0948f4c81c650bc`), serve solo il file `model.safetensors` da quel repository

Installa le dipendenze ed esegui il programma con `./test.sh` — costruirà ed eseguirà immediatamente.

Gira anche su GPU AMD tramite ROCm/HIP. Passa `-DUSE_HIP=ON` a CMake e compila con `hipcc` contro `hipBLAS` invece di `nvcc` e `cuBLAS`; i sorgenti CUDA vengono riusati così come sono tramite un header di compatibilità `src/cuda_to_hip.h`. Scegli l'architettura della tua GPU con `-DCMAKE_HIP_ARCHITECTURES` (per esempio `gfx90a` per MI200, `gfx1100` per RDNA3, `gfx1201` per RDNA4) — non è hardcoded, quindi impostala in base alla tua scheda:

```
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm -G Ninja
cmake --build build
```

`-DCMAKE_PREFIX_PATH=/opt/rocm` permette a CMake di trovare i pacchetti hip e hipBLAS; ometti questa opzione se `/opt/rocm/bin` è già nel tuo `PATH`, oppure cambiala se ROCm si trova altrove. L'autore ha testato il percorso AMD su gfx90a, gfx1100 e gfx1201. La build di default (senza `-DUSE_HIP`) resta invariata e continua a puntare a NVIDIA tramite CUDA.

Se non riesci a compilare o eseguire e la tua IA di fiducia non riesce ad aiutarti, apri una Issue su GitHub — l'autore proverà ad aiutare. Assicurati di fornire tutto il contesto utile.

## Safetensors e il tuo modello

Prima cosa da fare: scaricare un LLM da usare per l'inferenza. L'autore sceglie Llama 3.2 1B Instruct, perché è facile, piccolo, ottimizzato per il dialogo e abbastanza buono per noi. Dal punto di vista di chi costruisce un server di inferenza, il modello è solo un singolo file che contiene i pesi.

Il modello è in [formato Safetensors](https://huggingface.co/docs/safetensors/index). Esistono altri formati, come [Pickle](https://docs.python.org/3/library/pickle.html) e [Parquet](https://parquet.apache.org/docs/file-format/). Safetensors è semplicemente molto popolare e diffuso, e il modello scelto è ospitato in Safetensors.

Fermiamoci un momento a capire il formato Safetensors prima di andare avanti.

Un file safetensors è composto da 3 sezioni, sempre in quest'ordine: dimensione dell'header, header e dati dei tensori. La dimensione dell'header è sempre 8 byte. Questi 8 byte sono un intero senza segno a 64 bit, che dice quanti byte occupa l'header vero e proprio.

```cpp
std::ifstream safetensors_file("model.safetensors", std::ios_base::binary);
uint64_t header_size;
safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
```

L'header è un JSON che contiene informazioni su tutti i tensori dentro il file. JSON è solo un gruppo di coppie <chiave, valore>, dove la chiave è una stringa unica con il nome del tensore e il valore è un altro oggetto JSON, con informazioni su quel tensore. Ogni chiave in questo JSON è il nome di un tensore, tranne un'unica chiave chiamata `__metadata__`, probabilmente per informazioni aggiuntive quando necessario (non la useremo, le specifiche dicono che è una "chiave speciale per memorizzare una mappa testo-a-testo in forma libera"). Ogni valore è un JSON contenente tre chiavi — `dtype`, `shape` e `offsets`. `dtype` dice in che tipo di dato è memorizzato il tensore. `shape` dice le [dimensioni del tensore](https://en.wikipedia.org/wiki/Tensor#As_multidimensional_arrays) e `offsets` dice dove è memorizzato il tensore, all'interno della sezione dati dei tensori. Ogni `shape` è una lista di interi di lunghezza sconosciuta, e ogni valore `offsets` è un vettore di esattamente due interi. Il primo elemento dice dove inizia il tensore e l'ultimo dice dove finisce.

A questo punto affronti la prima decisione di design. Vuoi rendere l'architettura del tuo server indipendente dal modello, così può far girare qualsiasi modello arbitrario, purché tu implementi le operazioni che gli servono, oppure vuoi iniziare in modo semplice concentrandoti sul modello scelto?

Qualunque cosa tu decida, è sempre più facile sviluppare su un singolo modello e poi generalizzare, piuttosto che cercare di renderlo flessibile fin dall'inizio quando non sei sicuro di come apparirà il codice alla fine. Puoi sempre tornarci sopra e aggiornarlo quando scegli di farlo.

Se vuoi rendere il tuo server indipendente dal modello, devi allocare la memoria, impostare `shape` e tipo (`dtype`) dei dati del modello dinamicamente in base all'header Safetensors e implementare più operazioni, assicurandoti di coprire tutte le operazioni usate da tutti i modelli che vuoi supportare. Probabilmente dovresti comunque fornire dei progetti dell'architettura dei modelli, perché il file Safetensors non ti dice quale operazione eseguire con quei dati, in che ordine, ecc. L'autore non è sicuro di quale sia l'approccio ottimale, ma se lo scopri, da solo o leggendo il codice di vLLM/[TensorRT](https://github.com/NVIDIA/TensorRT)/altri, condividi pure le tue scoperte.

L'autore assume di codificare il server solo per l'architettura di Llama 3.2 1B Instruct. Ecco un dump dell'oggetto `LlamaForCausalLM` di [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) con il modello `meta-llama/Llama-3.2-1B-Instruct` caricato. Possiamo ispezionare quali operazioni dobbiamo implementare e quale forma e tipo di dati dobbiamo usare:

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
```

Analizziamolo.

Prima di tutto, da qui non sappiamo né l'ordine delle operazioni né il tipo di dato. Ma! La [scheda del modello](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) su Hugging Face ci dice che i pesi sono in formato [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format). Torneremo presto su questo formato.

Dobbiamo capire l'ordine delle operazioni per sapere come codificarlo. Sebastian Raschka ha una galleria di architetture LLM che mostra bene come sono organizzate le operazioni — vedi [qui (quella a sinistra)](https://magazine.sebastianraschka.com/i/168650848/61-qwen3-dense).

Guardando il diagramma di Sebastian, vediamo che l'ordine delle operazioni in Llama 3.2 1B è così:

1. Invia del testo al modello
2. Trasformalo in token (un nuovo concetto, non ancora menzionato)
3. Recupera un embedding per ogni token
4. 16 blocchi transformer, chiamati anche layer, che consistono in:

- RMS Norm
- Connessione residua
- Masked grouped-query attention, che consiste in:
  * Proiezione Q
  * Proiezione K
  * Proiezione V
  * RoPE con proiezione Q
  * RoPE con proiezione K
  * Attention
  * Score di attention
  * Maschera causale
  * Softmax
  * Connessione residua
  * Score di attention con proiezione V
- Proiezione O (proiezione di output)
- Somma della connessione residua
- RMS Norm
- [Feed forward](https://en.wikipedia.org/wiki/Feedforward_neural_network) (come nelle prime reti neurali, [Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)), che consiste in:
  * Proiezione gate, primo layer lineare
  * Proiezione up, secondo layer lineare
  * Funzione di attivazione [SiLU](https://arxiv.org/pdf/1702.03118), simile a [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) ma assomiglia più a una sigmoide
  * Proiezione down, terzo layer lineare
  * Somma della connessione residua

5. RMS Norm
6. Output lineare
7. Argmax

Dopo questi passaggi, dovremmo ottenere il nostro primo token predetto dal modello linguistico eseguito sul nostro server.

Non preoccuparti se alcune di queste operazioni non ti sono ancora familiari. Le capirai in modo viscerale man mano che procediamo nel corso.

## Come funzionano i numeri floating-point e perché usiamo bfloat16

Ricordiamoci cosa vogliamo ottenere. Vogliamo caricare un modello. Conosciamo già la struttura del file con un modello, un file Safetensors. Conosciamo l'architettura del nostro modello di riferimento. Abbiamo verificato che i pesi del modello sono memorizzati in tipo BF16. Prendiamoci un momento per pensare a questo tipo e ai numeri floating-point in generale.

Tutto sul computer è binario, alla fine. Nei modelli ML, i pesi quasi mai sono numeri interi. Sono piuttosto numeri reali. E i computer sono binari. Significa che bisognava capire come rappresentare numeri reali nei linguaggi di programmazione, in modo efficiente in termini di memoria — cioè, poter impacchettare molta informazione in poco spazio.

Il tipo semplice, `float16` (IEEE 754), è composto da tre sezioni: segno, esponente e frazione, per un totale di 16 bit:

```
[ segno | esponente | frazione            ]
[ 0     | 0 1 0 0 1 | 0 0 1 0 0 0 0 0 0 0 ]
```

Formula: $(-1)^{segno} \times 2^{esponente-bias} \times (1.frazione)$

Il segno è 1 bit (0 = positivo, 1 = negativo). L'esponente è 5 bit e controlla la grandezza del numero. La frazione è 10 bit e controlla il numero usato per spostare il punto decimale (da cui il nome "floating" — mobile). Il `1.` prima della frazione è implicito — un trucco per guadagnare un bit di precisione senza memorizzarlo esplicitamente. Il `bias` (15 per float16) serve a poter rappresentare anche numeri più piccoli di 1, permettendo esponenti effettivamente negativi.

Il nostro modello di riferimento usa però **bfloat16** (BF16), non float16. bfloat16 occupa anch'esso 16 bit totali, ma ha un esponente più lungo (8 bit, la stessa dimensione del float32) — quindi rispetto a float16, bfloat16 scambia 3 bit di frazione per guadagnare 3 bit di esponente. La frazione si riduce quindi a 7 bit. In sintesi: bfloat16 è 1 bit di segno, 8 bit di esponente, 7 bit di frazione.

Perché conta bfloat16? Ha la stessa dimensione di float16 (metà di un float a piena precisione), ma allo stesso tempo ha lo stesso esponente di float32 — al costo di una frazione più piccola. L'industria spesso lo sceglie per l'inferenza, perché è meno soggetto a problemi di range (overflow/underflow) e allo stesso tempo la perdita di precisione (dovuta alla frazione più piccola) è un compromesso accettabile nell'inferenza LLM (i risultati empirici lo dimostrano).

> A proposito, i float a 32 bit si chiamano float a precisione singola. E poi senti parlare di doppia precisione. Potresti pensare — quindi ottengo DUE punti mobili nello stesso numero? Purtroppo doppia precisione significa solo che il numero è più grande (64 bit).

Ora hai tutti i prerequisiti per caricare un modello nel tuo motore di inferenza e iniziare a fare cose interessanti.

## Memoria GPU e CPU

Questo capitolo potrebbe sembrare un po' casuale, ma è utile se sei ancora agli inizi con la programmazione GPU. I dati possono vivere sull'host o sul device.

**Host** è il tuo PC con la CPU. Ha memoria grande e lenta — DRAM — quella che compri e inserisci nella scheda madre. Recentemente molto costosa, a causa della frenesia AI del big tech. La DRAM è separata dalla CPU. La CPU ha anche una propria memoria on-chip piccola e veloce — SRAM.

**Device** è la tua GPU. Ha anch'essa memoria grande e lenta — HBM, VRAM — ma è saldata sulla scheda e non la puoi espandere facilmente come fai con la DRAM nel tuo PC. Ha memoria piccola e veloce — SRAM — che puoi usare se definisci variabili `__shared__`.

La GPU non può accedere alla tua DRAM, quindi prima di eseguire qualsiasi calcolo sulla GPU, devi copiarci i dati. Il flusso tipico può essere così:

1. Crea una variabile sulla CPU
2. Scrivi sulla variabile sulla CPU
3. Calcola la dimensione di memoria occupata dalla variabile sulla CPU, oppure trova la dimensione massima in altro modo
4. Moltiplica la dimensione calcolata per la dimensione del tipo di variabile
5. Alloca la memoria sulla GPU
6. Copia la variabile dalla CPU alla GPU
7. Ora puoi usare questi dati nei tuoi calcoli GPU

Idealmente vuoi allocare meno memoria possibile, riusarla il più possibile, e copiare i dati il meno possibile.

Esempio: sapere quali token sono attualmente attivi durante l'inferenza. Servono sia su CPU che GPU.

```cpp
std::vector<int> active_tokens;
active_tokens.push_back(token);

// BATCH_SIZE = numero massimo di token attivi possibile
int *gpu_active_tokens;
cudaMalloc(&gpu_active_tokens, BATCH_SIZE * sizeof(int));

//               destinazione,               sorgente,           dimensione dati da copiare, direzione della copia
cudaMemcpy(gpu_active_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);

// Ora puoi usare questi dati quando invochi i kernel GPU:
embeddingGatherDecode(gpu_active_tokens, num_active_slots, hidden_state, weights.embed_tokens);
```

## Inferenza di un singolo token

Iniziamo a lavorare sull'inferenza. Puoi chiudere questo corso ora e iniziare a scrivere codice, basandoti sulla sequenza di operazioni delineata nella sezione [Safetensors e il tuo modello](#safetensors-e-il-tuo-modello), oppure continuare a leggere per riempirti la testa di cose utili.

> **Nota sull'apprendimento con gli LLM (dell'autore, 2026)**: il processo di apprendimento può essere pensato come questo ciclo: 1. Capisci più o meno cosa vuoi costruire. 2. Descrivi il tuo modello mentale a un chatbot, chiedigli di individuare i tuoi punti ciechi, errori nel tuo ragionamento, colmare le lacune nella tua conoscenza con informazioni su misura. 3. Leggi la risposta, prova a interiorizzarla e aggiorna il tuo modello mentale. 4. Ripeti finché non ci sono più errori. 5. Inizia a scrivere codice, continua finché non sei bloccato. 6. Quando sei bloccato, torna al punto 2. L'autore sottolinea l'importanza di distinguere tra sforzo di comprensione (dove avviene davvero l'apprendimento — scomporre, capire il meccanismo, il flusso dei dati, le relazioni, fare errori) e sforzo sprecato (recupero meccanico di dati, azioni che consumano tempo senza insegnarti nulla di nuovo) — quest'ultimo ha senso delegarlo a un LLM, il primo no.

## Tokenizzazione

Una volta caricato il modello e mappati i pesi, bisogna leggere l'input dell'utente, il prompt, e trasformarlo da testo in qualcosa che il modello capisce — i token. Tutti gli LLM mainstream usano token, non parole o caratteri.

Per trasformare il testo in una sequenza di token, serve un tokenizer. Useremo un tokenizer esistente, che produce token corrispondenti al dizionario di Llama 3.2 1B (vedi `python/tokenizer.py` nel repo, che usa un tokenizer di Hugging Face).

Quello che devi ricordare è che prende un testo e produce una sequenza di token (interi), che rappresentano il tuo testo come vettore di interi. E l'LLM ha bisogno del tuo testo come questo vettore di interi.

## Embedding

Il tuo testo è tradotto in token e li dai in pasto al tuo server di inferenza LLM. I token sono più che altro indici, ma non sono i dati su cui lavorerà davvero il tuo LLM. I modelli linguistici di grandi dimensioni sanno come mappare ogni token a un vettore, dove ogni token ha la stessa lunghezza di vettore, ma valori diversi. Questi vettori si chiamano embedding. Incorporano il significato di un token. Poi dai in pasto una lista di token all'LLM, che recupera un embedding per token, dove i token funzionano come indici che dicono al modello quale embedding (vettore) recuperare dai suoi pesi. Nel nostro caso, ogni embedding ha lunghezza 2048. Quindi per 5 token in input, ottieni 5 vettori di lunghezza 2048, che insieme formano una matrice di dimensione (5, 2048). Conosciamo già il tipo di ogni numero in questi vettori di embedding — è bfloat16.

Questo potrebbe essere il primo kernel CUDA da scrivere in questo corso. Il tuo compito è recuperare gli embedding per tutti i token in input.

```cpp
std::vector<int> input_tokens = {678, 264, 1933, 13};
constexpr int MAX_PROMPT_LEN = 512;

int *gpu_input_tokens;
cudaMalloc(&gpu_input_tokens, MAX_PROMPT_LEN * sizeof(int));

//              destinazione,              sorgente,                              dimensione, direzione
cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
```

## Ingegneria di kernel CUDA — embedding

> Esistono risorse molto migliori di quanto l'autore possa produrre, quindi per imparare CUDA, consulta la [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html). L'autore accenna solo brevemente le basi qui.

I kernel sono funzioni eseguite su una GPU. Lanci la stessa funzione più volte. Ogni funzione lanciata è un thread separato. Eseguono lo stesso codice. Ricevono parametri leggermente diversi, come l'indice di un thread. I thread sono raggruppati in block. Quando lanci un kernel CUDA, definisci quanti block vuoi invocare e quanti thread ci sono in ogni block. I thread sono anche raggruppati in warp. Ogni warp ha 32 thread. Quindi, se lanci il tuo kernel e definisci che deve eseguire 5 block e ogni block deve eseguire 64 thread, ogni block esegue 2 warp, 32 thread ciascuno.

Quando si scrivono kernel CUDA, molto sforzo va nel pensare alla memoria — dalla prospettiva del thread: "quali dati dovrei processare?", "dove dovrei scrivere i risultati?". In pratica significa capire l'indice dei dati in input da leggere e dove scrivere l'output. I tuoi strumenti principali sono variabili built-in, come `threadIdx`, `blockIdx` e `blockDim`. Ogni thread ha i propri valori per queste variabili. Questo rende possibile eseguire lo stesso calcolo in parallelo. Questo approccio si chiama [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads).

Sappiamo che per ogni token in input, vogliamo recuperare un embedding composto da 2048 numeri bfloat16. Il primo approccio: abbiamo N token da recuperare, e ogni token ha bisogno di recuperare 2048 numeri. Possiamo lanciare N block — un block per token — e 2048 thread in ogni block, così ogni thread recupera esattamente un numero.

```cpp
__global__ void embeddingGatherKernel()
{
}
```

Invocazione:

```cpp
embeddingGatherKernel<<<num_input_tokens, 2048>>>();
```

Serve un output dove scrivere. Allochiamo memoria per 2048 numeri bfloat16 per N token:

```cpp
__nv_bfloat16* input_embeddings;
cudaMalloc(&input_embeddings, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * 2048);
```

Passiamo i puntatori al kernel:

```cpp
embeddingGatherKernel<<<num_input_tokens, 2048>>>(gpu_input_tokens, input_embeddings, weights.embed_tokens);

__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
```

Pensiamo al kernel dalla prospettiva di un singolo thread. Se avessimo un solo token, potremmo mappare 1:1 `threadIdx.x` all'indice di `input_embeddings`. Per recuperare il numero corretto dell'embedding per questo token, dobbiamo moltiplicare il token per 2048 — `gpu_input_tokens[0] * 2048` — per ottenere l'indice del primo numero dell'embedding del token, poi spostare l'indice del valore corrente di `threadIdx.x`:

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    input_embeddings[threadIdx.x] = embed_tokens[gpu_input_tokens[0] * 2048 + threadIdx.x];
}
```

Ora generalizziamo per più di un token in input. Ogni token risulta in un embedding di dimensione 2048. Moltiplichiamo l'indice del block per 2048 e sommiamo `threadIdx.x` per ottenere una posizione nell'embedding del token attualmente processato (e indice_token == indice_block):

```cpp
int workIndex = threadIdx.x + blockIdx.x * 2048;
```

Non possiamo più hardcodare il primo token `gpu_input_tokens[0]`. Dato che indice_token == indice_block, sostituiamo `[0]` con `blockIdx.x`:

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
}
```

Creiamo una funzione wrapper sopra questa invocazione del kernel, così puoi eseguirla dal tuo codice C++:

```cpp
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    embeddingGatherKernel<<<num_input_tokens, 2048>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens);
}
```

Tutto sembra a posto! Lanciamolo. E poi?

...

[È una trappola!](https://www.youtube.com/watch?v=4F4qzPbcFiA) Il massimo di thread per block è 1024, almeno per la maggior parte delle GPU NVIDIA che potresti avere a casa.

Cosa possiamo fare? Possiamo lanciare 2 volte più block, oppure processare 2 numeri in ogni thread invece di uno solo. L'autore sceglie la seconda opzione — probabilmente più veloce, non richiede più thread e non richiede sincronizzazione tra thread. Con 1024 thread massimi per block e 2048 numeri nell'embedding, vogliamo processare due numeri per thread. L'opzione più semplice: ogni thread processa il proprio numero e il numero nella stessa posizione nella seconda metà dell'embedding, quindi indice del thread corrente + 1024.

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
    input_embeddings[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
}
```

Eseguilo. Questa volta funzionerà.

Complimenti, hai finito il tuo primo kernel CUDA!

Torniamo dal calcolo e dalla programmazione a basso livello al semantico/significato di quello che abbiamo appena fatto. Nota che, sebbene tutti questi embedding recuperati per i token in input abbiano un significato codificato al loro interno, non si "conoscono" tra loro. Non conoscono la propria posizione all'interno del testo fornito in input. Non sanno da quali token sono circondati. Non conoscono la storia della conversazione, ecc. Questa comprensione verrà costruita e memorizzata come proiezioni K e V.

## RMSNorm e riduzione parallela in CUDA

Dopo aver recuperato gli embedding per i nostri token, è il momento di [RMSNorm](https://arxiv.org/abs/1910.07467). A differenza del recupero degli embedding, è la prima operazione che gira nei layer. Il nostro modello, Llama 3.2 1B, ha 16 layer. RMSNorm prende gli embedding recuperati e — usando i pesi del modello per una rms norm `weights.input_layernorm[layer]` — esegue la funzione RMSNorm. RMSNorm è un'operazione che modifica tutti i numeri in un embedding. Per farlo, prima deve vedere tutti gli elementi e calcolare la loro somma di [radice quadratica media](https://en.wikipedia.org/wiki/Root_mean_square).

Formula (dal paper):

$$ \text{normalizzato}_i = \frac{a_i}{\text{RMS(a)}} \text{, dove RMS(a)}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a^2_i} $$

Per calcolare `RMS(a)` serve passare per tutti i numeri, quindi serve sincronizzazione tra thread. Normalizziamo ogni embedding separatamente, quindi lanciamo ancora tanti block quanti token in input. Con 1024 thread massimi per block e 2048 numeri, usiamo lo stesso trucco delle due letture per thread. Serve un vettore temporaneo dove ogni thread scrive il quadrato del proprio numero — questa tecnica si chiama [riduzione parallela](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (tree reduction). Si usa `__shared__` per indicare che il vettore è condiviso tra tutti i thread di un block.

> Nota sulla stabilità numerica: i dati sono in formato bfloat16, con mantissa piccola (7 bit). Per non perdere precisione nei calcoli interni al kernel (come il quadrato di ogni numero), si usa `float` come tipo di calcolo intermedio, castando indietro a `__nv_bfloat16` solo alla fine.

```cpp
__shared__ float rms_vector[1024];
int workIndex = threadIdx.x + blockIdx.x * 2048;
rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
__syncthreads();
```

Ogni thread in un block memorizza il quadrato del proprio numero in input sommato al quadrato del numero spostato di 1024 (coprendo la seconda metà del vettore di 2048, che non possiamo coprire lanciando 2048 thread in un block, per il limite CUDA di 1024 thread massimi per block). `rms_vector[0]` contiene 2 quadrati su 2048 numeri dell'embedding di input attualmente processato (per il token corrente) — indici `0` e `1024`.

Primo step di riduzione: ogni secondo thread aggiunge il numero successivo in `rms_vector`.

```cpp
if (threadIdx.x % 2 == 0) {
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1];
}
__syncthreads();
```

`%` è l'operatore modulo (divisione con resto). Ora `rms_vector[0]` memorizza i quadrati dei numeri agli indici `0`, `1024` e `1`. Sappiamo che `1` conteneva già `1` e `1025`, quindi `rms_vector[0]` ha in realtà questi 4 quadrati: `0`, `1`, `1024` e `1025`.

Aumentiamo il "salto" di 2, così ogni 4° elemento aggiungerà un elemento distante 2 indici da sé:

```cpp
if (threadIdx.x % 4 == 0) {
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 2];
}
__syncthreads();
```

Il processo continua fino a fare modulo per 1024 su `threadIdx.x`. Versione compatta con ciclo:

```cpp
__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights)
{
    __shared__ float rms_vector[1024];
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
    __syncthreads();
    // tree reduction
    for (int i = 1; i < 1024; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0)
        {
            rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5); // = RMS(a), con epsilon anti divisione-per-zero
    }
    __syncthreads();

    output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
    output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
}
```

Nota sull'epsilon: se `RMS(a)` fosse 0, la divisione produrrebbe NaN o infinito, corrompendo l'inferenza. Il modello di riferimento ha un parametro epsilon (`eps=1e-05`) proprio per prevenire questo.

Complimenti per aver finito il tuo kernel RMSNorm CUDA!

## RoPE

Nel modello di riferimento, l'operazione successiva a RMSNorm è [RoPE](https://arxiv.org/pdf/2104.09864), un modo di codificare la posizione dei token nello stato nascosto (embedding). Una descrizione molto accessibile del positional encoding con RoPE è [qui, di Christopher Fleetwood](https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding).

```cpp
__global__ void ropeKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    if (2 * threadIdx.x + 1 + blockIdx.x * proj_dim < num_tokens * proj_dim)
    {
        // TODO: precalcolare theta, angoli e magari i valori sin/cos, riusandoli su tutte le invocazioni del kernel
        int double_i = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
        float angle = blockIdx.x * theta;
        __nv_bfloat16 prev_2i = input[2 * threadIdx.x + blockIdx.x * proj_dim];
        __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim];
        input[2 * threadIdx.x + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * cos(angle) - (float)prev_2i_1 * sin(angle));
        input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * sin(angle) + (float)prev_2i_1 * cos(angle));
    }
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: serve parametrizzarlo per usarlo sia per q_proj che k_proj (1024 per q_proj, 512 per k_proj)
void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim);
}
```

## Connessioni residue

È una tecnica semplice usata in molti modelli di deep learning diversi, dove sommi gli input agli output, così i tuoi dati in input non vengono mai persi del tutto. Si fa per stabilità nel training. Dal nostro punto di vista è solo la somma elemento per elemento di due vettori della stessa dimensione.

```cpp
__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input[workIndex] = input[workIndex] + input_embeds[workIndex];
    input[workIndex + 1024] = input[workIndex + 1024] + input_embeds[workIndex + 1024];
}

// (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens)
{
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
}
```

## cublasGemmEx

[Moltiplicazione di matrici](https://en.wikipedia.org/wiki/Matrix_multiplication) è una delle operazioni principali usate nel deep learning, in particolare negli LLM. Una matrice è una tabella di numeri, con righe e colonne. Una singola riga o colonna si chiama vettore. La moltiplicazione di matrici usa due matrici, A e B, in input, e produce la matrice C in output. A ha dimensioni (M, K), B ha dimensioni (K, N) — condividono la dimensione K. Moltiplicando A per B ottieni una nuova matrice C con dimensioni (M, N):

$$A (M,K) \times B(K,N)=C(M,N)$$

Ogni elemento di C ($c_{ij}$) è un [prodotto scalare](https://en.wikipedia.org/wiki/Dot_product) tra la riga i-esima di A e la colonna j-esima di B.

La moltiplicazione di matrici avviene quando si calcolano attention e proiezioni Q, K, V. L'hardware più diffuso per calcolarla efficientemente sono le GPU NVIDIA, che forniscono una libreria importante, [cuBLAS](https://developer.nvidia.com/cublas), per calcoli di algebra lineare ad alte prestazioni, inclusa la moltiplicazione di matrici, tramite la funzione [cublasGemmEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex).

## Il trucco di trasposizione column-major → row-major

**TL;DR: se i tuoi dati sono in formato row-major e usi cuBLAS, imposta il flag di trasposizione a `CUBLAS_OP_T` per le matrici non ancora trasposte, e `CUBLAS_OP_N` per le matrici già trasposte nella tua formula.**

Il problema con `cublasGemmEx` è che si aspetta le matrici in [formato column-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order). Gli LLM, come Llama 3.2 1B Instruct, sono distribuiti in formato row-major.

Non serve modificare il formato dei dati per usare le funzioni di moltiplicazione di matrici di cuBLAS. Grazie a queste proprietà:

$$[A^T]_{ij}=[A]_{ji} \qquad C^T=B^T \times A^T \qquad (A^T)^T=A$$

"$^T$" significa trasporre la matrice — trasformare colonne in righe e viceversa. Quando memorizzi la matrice in formato row-major, e cuBLAS la legge in formato column-major, è l'equivalente di averla trasposta.

Esempio: vogliamo calcolare $C = A \times B$, dove A ha dimensioni (5, 2048) e B ha dimensioni (512, 2048). Vogliamo C di dimensione (5, 512). Le dimensioni di A e B sono incompatibili così come sono — serve trasporre B: $C = A \times B^T$. Ora le dimensioni tornano: $A(5,2048) \times B(2048, 512) = C(5, 512)$.

Ma cuBLAS si aspetta A e B in formato column-major. Row-major trasposto ci dà column-major. Trasponiamo la formula $C = A \times B^T$, usando la proprietà $C^T=B^T \times A^T$: $C^T = (B^T)^T \times A^T$. Semplificando $(B^T)^T$ a $B$: $C^T = B \times A^T$. Verifica dimensioni: $B (512, 2048) \times A^T(2048, 5) = C^T(512, 5)$ — che è l'inverso di quello che volevamo (5, 512), ma ricorda che parliamo di $C^T$. La $C$ vera, output di `cublasGemmEx`, non è trasposta, quindi la dimensione finale è corretta (5, 512).

```cpp
cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, KV_DIM, num_active_slots, EMBEDDING_LENGTH, &k_proj_alpha, weights.w_k[layer], CUDA_R_16BF, EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH, &k_proj_beta, k_proj_batched_buffer, CUDA_R_16BF, KV_DIM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

Perché diciamo a cuBLAS di trasporre la prima matrice $B$, se la formula derivata è $C^T=B \times A^T$? Dal punto di vista di cuBLAS, la nostra $B$ row-major è vista come $B^T$ trasposta, perché cuBLAS la legge come se fosse column-major. Quindi dobbiamo dirgli di trasporla, per riottenere la $B$ che abbiamo derivato. Analogamente, dato che abbiamo derivato che il secondo argomento dovrebbe essere $A^T$, e cuBLAS legge la $A$ row-major come una $A^T$ column-major, non trasponiamola di nuovo, perché è già come volevamo fornirla a `cublasGemmEx`.

## Prefill vs decode

Un fatto interessante sull'inferenza LLM è che il processo non è esattamente lo stesso per il primo token predetto rispetto a tutti i successivi. Per predire il primo token, devi processare tutti i token in input. I token in input sono il prompt dell'utente o la cronologia della chat. Tutto il calcolo necessario per ottenere il primo token predetto si chiama *prefill*. Tutto ciò che succede dopo si chiama *decode*.

La maggior parte del calcolo, come proiezione Q, attention, score di attention, feed-forward (MLP), viene scartato non appena passato all'operazione successiva — sia in prefill che in decode. Il modello mentale utile è che l'unica cosa che preservi ad ogni fase dell'inferenza LLM è la proiezione K, la proiezione V e qual è l'ultimo token generato. Tutto qui. Ci sono implicazioni interessanti — potresti fermare l'inferenza, copiare le tue proiezioni K e V e l'ultimo token generato, riavviare il server, ricaricarle nel server e usare l'ultimo token generato come input, ottenendo la stessa identica predizione del token successivo dell'istanza originale del server.

## Perché esiste il KV cache

Possiamo riusare alcune parti dei risultati del calcolo per predire i token successivi. Non è obbligatorio riusarli, ma non cambiano, quindi ricalcolarli continuamente è puro spreco. L'unico dato che va avanti nel calcolo è la proiezione K, V e l'ultimo token generato. Se generiamo 1 token alla volta, K e V sono vettori di bfloat16 e l'ultimo token generato è un singolo intero. Con il batching, K e V sono matrici e gli ultimi token generati sono un vettore di interi.

Quando processiamo un token, sia in prefill che in decode, dal punto di vista dei dati preservati (proiezioni K, V e ultimo token generato) appare così:

```
0. ...
1. Calcola proiezione K usando l'ultimo token generato
2. Memorizzala
3. Calcola proiezione V usando l'ultimo token generato
4. Memorizzala
5. ...
6. Usa tutte le proiezioni K e V per calcolare l'attention
7. ...
8. Genera nuovo token
9. Memorizzalo come ultimo token generato
```

Se non memorizzassimo le proiezioni K e V per il token corrente, dovremmo ricalcolare tutte le proiezioni K e V per il token corrente e tutti i precedenti prima di poter calcolare l'attention per il token corrente. Di nuovo, puro spreco. Ecco perché memorizziamo le proiezioni K e V. È solo una registrazione di tutte le proiezioni K e V precedenti. Non la modifichi durante l'inferenza LLM. Ci aggiungi solamente, ad ogni token processato. Il nome di questo storage di proiezioni K e V è KV cache.

## Attention

L'attention è una parte importante dell'inferenza LLM. È dove fai molta moltiplicazione di matrici usando le proiezioni Q, K e V calcolate prima. La formula base per lo scaled dot-product attention, dal paper [Attention is all you need](https://arxiv.org/pdf/1706.03762):

$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

```cpp
for (int i = 0; i < NUM_Q_HEADS; ++i)
{
    int k_head_idx = i / GQA_Q_TO_K_RATIO; // i / 4 <- significa che 4 head Q usano la stessa 1 head K
    __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
    __nv_bfloat16 *k_head = k_proj[layer] + k_head_idx * HEAD_DIM;
    __nv_bfloat16 *attn_score_head = attn_scores + input_tokens.size() * input_tokens.size() * i;

    cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    input_tokens.size(),
                                                    input_tokens.size(),
                                                    HEAD_DIM,
                                                    &attn_alpha,
                                                    k_head,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    q_head,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &attn_beta,
                                                    attn_score_head,
                                                    CUDA_R_16BF,
                                                    input_tokens.size(),
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
}

// maschera causale, softmax e poi score di attention * V
        // attn scores * V
        // (32, num_tok, num_tok) * (num_tok, 512)
        // GQA - 4 head Q condividono 1 head V
        // dim attn_scores (32, num_tok, num_tok)
        // dim head attn_scores (num_tok, num_tok)
        // dim V (num_tok, 512)
        // NUM_V_HEADS è 8 -> 512 / 8 = 64
        // dim V_head (num_tok, 64)
        // dim output head: scores head * V head -> (num_tok, num_tok) * (num_tok, 64) = (num_tok, 64)
        // in totale 32 output head: quindi (num_tok, 64 * 32) = (num_tok, 2048)
for (int i = 0; i < NUM_Q_HEADS; ++i)
{
    int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO; // GQA, 4 head Q per 1 head V
    __nv_bfloat16 *attn_scores_head = attn_scores + i * input_tokens.size() * input_tokens.size();
    __nv_bfloat16 *v_head = v_proj[layer] + v_head_idx * HEAD_DIM;
    __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

    cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    HEAD_DIM,
                                                    input_tokens.size(),
                                                    input_tokens.size(),
                                                    &attn_scores_v_alpha,
                                                    v_head,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    attn_scores_head,
                                                    CUDA_R_16BF,
                                                    input_tokens.size(),
                                                    &attn_scores_v_beta,
                                                    output_attn_scores_head,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
}
```

## GQA

A differenza della multi-head attention, uno dei primi metodi di attention, nella [group-query attention (GQA)](https://arxiv.org/pdf/2305.13245) più head di query condividono la stessa head di key e value. Nel nostro caso: 4 head di query usano la stessa 1 head di key e value. Vedi il codice con il calcolo delle head K/V sopra.

## SiLU

[SiLU](https://arxiv.org/pdf/1702.03118) è una funzione di attivazione usata nel modello di riferimento. Introduce "non-linearità" in un modello — significa che i pesi possono essere azzerati quando non servono. Aiuta nel training dei modelli. Quasi tutti i modelli di machine learning ce l'hanno nella loro architettura, anche i più semplici multi-layer perceptron. SiLU è simile a ReLU, ma quando i valori negativi si avvicinano a 0, non vengono azzerati, ma ottengono invece un piccolo valore negativo.

```cpp
__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b)
{
    int workIndex = threadIdx.x + blockIdx.x * 8192;
    for (int i = 0; i < 8192; i += 1024)
    {
        a[workIndex + i] = (__nv_bfloat16)((float)a[workIndex + i] * (1 / (1 + expf(-(float)a[workIndex + i]))) * (float)b[workIndex + i]);
    }
}

// in-place, sovrascrive a
void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens)
{
    siluKernel<<<num_tokens, 1024>>>(a, b);
}
```

## Softmax

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) è una funzione che normalizza tutti gli elementi in un vettore. Questa è la prima versione, "sequenziale", del softmax. Deriveremo e implementeremo la versione "online" più avanti.

$$ \sigma(v)=\frac{e^{v_i}}{\sum_{j=1}{K}e^{v_j}} $$

Il modo per implementarlo è molto simile a RMSNorm implementato prima.

```cpp
__global__ void softmaxKernel(__nv_bfloat16 *input, int num_tokens)
{
    // softmax per head
    __shared__ float row[1024]; // row[0] conterrà il valore massimo dopo il ciclo
    __shared__ float max_val;
    // trova il massimo della riga per sottrarlo, per stabilità numerica
    int workIndex = blockIdx.x * num_tokens + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;
    __syncthreads();

    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0];
    }
    __syncthreads();

    // trasforma in exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // ora calcoliamo la somma numericamente stabile, stesso pattern - tree reduction
    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
}

// input sono gli score di attention mascherati (NUM_Q_HEADS, num_tok, num_tok)
void softmax(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
}
```

## Maschera causale

Ogni token può fare attention solo verso i token precedenti.

```cpp
__global__ void causalMaskKernel(__nv_bfloat16 *input, int num_tokens)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS)
    {
        return;
    }

    int column = threadIdx.x;
    int row = blockIdx.x % num_tokens;
    if (column > row)
    {
        input[blockIdx.x * num_tokens + threadIdx.x] = -HUGE_VALF;
    }
}

void causalMask(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched";
        return;
    }

    causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
}
```

## Argmax

Scegli il token con il punteggio più alto.

```cpp
cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * VOCAB_SIZE, cudaMemcpyDeviceToHost);
max_token = (float)embed_proj_cpu[0];
max_token_idx = 0;
for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
{
    if ((float)embed_proj_cpu[token_idx] > max_token)
    {
        max_token = embed_proj_cpu[token_idx];
        max_token_idx = token_idx;
    }
}
std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;
```

## Feed forward network

Una [feed-forward network](https://en.wikipedia.org/wiki/Feedforward_neural_network) / rete completamente connessa / multi-layer perceptron. Tre layer lineari e SiLU.

```cpp
cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, 1, EMBEDDING_LENGTH,
              &gate_alpha, weights.mlp_gate_proj[layer], CUDA_R_16BF, EMBEDDING_LENGTH,
              rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH,
              &gate_beta, gate, CUDA_R_16BF, HIDDEN_DIM,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

// (1, 2048) * (2048, 8192) -> (1, 8192)
cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, 1, EMBEDDING_LENGTH,
              &up_alpha, weights.mlp_up_proj[layer], CUDA_R_16BF, EMBEDDING_LENGTH,
              rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH,
              &up_beta, up, CUDA_R_16BF, HIDDEN_DIM,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

silu(gate, up, 1);

down = buf_2048_2;
cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, EMBEDDING_LENGTH, 1, HIDDEN_DIM,
              &down_alpha, weights.mlp_down_proj[layer], CUDA_R_16BF, HIDDEN_DIM,
              gate, CUDA_R_16BF, HIDDEN_DIM,
              &down_beta, down, CUDA_R_16BF, EMBEDDING_LENGTH,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

## Riuso dei buffer

A volte, un buffer che allochi ha la stessa dimensione di un buffer che verrà usato più avanti nel codice. E a volte, quando il secondo buffer inizia a servire dopo che i dati del primo buffer non servono più (è già stato usato in un calcolo e non verrà più usato altrove), possiamo usare il primo buffer per scrivere i dati che scriveremmo nel secondo buffer. In questo modo, possiamo allocare meno memoria — riusiamo lo stesso buffer in due punti diversi. Possiamo farlo in sicurezza una volta confermato, tramite un'analisi del ciclo di vita, che i cicli di vita di questi due buffer non si sovrappongono. Vedi come funziona per `buf_2048_1` e `buf_2048_2` in `src/main.cpp` nel repo.

## Batching statico

Invece di processare una richiesta alla volta, processiamo N richieste. Pro: throughput maggiore (più richieste utente processate contemporaneamente). Contro: latenza più alta (tutti i prompt nel batch devono aspettare che il prompt più lungo finisca di essere processato, prima di essere restituiti all'utente). In pratica: in ogni punto dove si assumeva un singolo token processato alla volta, bisogna processarne multipli. Lo stesso vale per il prefill — per più prompt, ecc.

## Batching continuo

Risolve il problema di dover aspettare il prompt più lungo nel batch. Lo risolve avendo slot in un batch. Riempi i prompt negli slot. Una volta finita la generazione in un dato slot, il risultato viene restituito all'utente. Un prompt in attesa in coda viene selezionato per lo slot appena liberato. Il prompt passa per il prefill (tutti gli altri elementi nel batch aspettano finché non è finito). Una volta finito il prefill, il batching continua con tutti gli elementi del batch, incluso quello nuovo.

## Online softmax

*(sezione ancora in scrittura nell'originale — vedi i riferimenti: [Online Softmax: Cohere Labs talk - Slides](https://docs.google.com/presentation/d/1f4Ys4es1l8bfUFN1bJuy12c3zHIPelcNGgVgbyb1zac/edit) e [From Boltzmann-Gibbs distribution to numerically stable online softmax on GPU](https://jedrzej.maczan.pl/boltzmann_gibbs_to_softmax_on_gpu.pdf))*

---

> ⚠️ **AVVISO — Le tre sezioni seguenti non fanno parte del README originale di Maczan.**
>
> Nel repository originale, queste tre sezioni erano marcate solo con "Incoming!" (in arrivo) — l'autore non le aveva ancora scritte al momento in cui ho letto il repo. Quanto segue è la **mia interpretazione**, basata su come funziona PagedAttention nel paper originale di vLLM (Kwon et al., 2023) e su quanto abbiamo già discusso insieme nel nostro corso teorico (in particolare il Modulo 10 sul KV cache e il Modulo 17 sul confronto tra vLLM e motori "locali"). **Non è testo di Maczan.** Quando l'autore pubblicherà la sua versione ufficiale di queste sezioni nel repo, ti consiglio di confrontarla con quanto scritto qui — è probabile che la sua implementazione concreta (nomi di variabili, struttura esatta del kernel, scelte di layout specifiche per Llama 3.2 1B) differisca nei dettagli da questa spiegazione concettuale.

## PagedAttention

*(sezione integrata da Claude — vedi avviso sopra)*

Nel KV cache "ingenuo" che abbiamo implementato finora, ogni richiesta riserva un blocco di memoria **contiguo**, dimensionato sulla lunghezza massima di sequenza possibile (es. `MAX_PROMPT_LEN` + token generabili). Questo funziona bene per un motore che gestisce poche richieste alla volta (come `tiny-vllm`, pensato per l'apprendimento), ma diventa un problema serio quando servi molte richieste simultanee con lunghezze molto diverse tra loro — lo scenario tipico di un motore da produzione come vLLM.

Il problema è duplice:

1. **Spreco per allocazione eccessiva**: se prenoti sempre la lunghezza massima per ogni richiesta, ma la maggior parte delle richieste reali è molto più corta, sprechi enormi quantità di memoria GPU "prenotata ma inutilizzata".
2. **Frammentazione**: richieste che iniziano e finiscono in momenti diversi, con lunghezze finali imprevedibili, lasciano "buchi" di memoria difficili da riutilizzare in modo efficiente con allocazioni contigue di dimensione variabile.

**L'idea di PagedAttention**, dal paper di Kwon et al. (2023, UC Berkeley), prende in prestito il concetto di **paginazione della memoria virtuale** dai sistemi operativi: invece di un unico blocco contiguo per richiesta, il KV cache viene diviso in tante **pagine** (o "blocchi") di dimensione fissa e piccola (es. spazio sufficiente per 16 token). Ogni richiesta ottiene le pagine di cui ha effettivamente bisogno, man mano che la sequenza cresce — non tutte in anticipo — e queste pagine **non devono essere fisicamente contigue in memoria**.

Una **tabella dei blocchi** (block table) per ogni richiesta tiene traccia di quali pagine fisiche, in che ordine, compongono la sequenza logica di quella richiesta — esattamente come una tabella delle pagine di un sistema operativo mappa indirizzi virtuali a indirizzi fisici.

## Paged KV cache

*(sezione integrata da Claude — vedi avviso sopra)*

Concretamente, invece di un buffer K/V per richiesta dimensionato a `MAX_PROMPT_LEN`, allochi **un pool globale** di pagine fisiche di dimensione fissa, condiviso tra tutte le richieste attive:

```cpp
constexpr int PAGE_SIZE = 16;      // token per pagina
constexpr int NUM_PAGES = 4096;    // pagine totali nel pool (dimensionato in base alla VRAM disponibile)

// Pool fisico: NUM_PAGES pagine, ciascuna con spazio per PAGE_SIZE token di K e V
__nv_bfloat16 *kv_cache_pool_k; // dimensione: NUM_PAGES * PAGE_SIZE * KV_DIM
__nv_bfloat16 *kv_cache_pool_v;

// Per ogni richiesta attiva, una tabella che mappa "posizione logica nella sequenza"
// a "quale pagina fisica del pool" (block table)
struct RequestBlockTable {
    std::vector<int> physical_page_ids; // physical_page_ids[i] = pagina fisica per il blocco logico i
};
```

Quando una richiesta genera un nuovo token e la sua pagina corrente è piena, il gestore di memoria (l'equivalente del "page allocator" di un sistema operativo) assegna una nuova pagina libera dal pool condiviso e la aggiunge alla block table di quella richiesta — non serve spostare o ricopiare le pagine già scritte, esattamente come non serve deframmentare la RAM quando un processo alloca nuova memoria virtuale.

Questo risolve entrambi i problemi visti sopra: **niente spreco per over-allocation** (allochi pagine solo quando servono davvero) e **niente frammentazione dannosa** (le pagine libere nel pool possono essere assegnate a qualsiasi richiesta, indipendentemente dalla loro posizione fisica).

Un dettaglio in più, utile anche in produzione: se due richieste condividono un prefisso comune (es. lo stesso system prompt), possono **condividere le stesse pagine fisiche** per quel prefisso, con un contatore di riferimenti (reference counting) — un altro parallelo diretto con il copy-on-write dei sistemi operativi.

## Kernel CUDA per Paged Attention

*(sezione integrata da Claude — vedi avviso sopra)*

Il kernel di attention deve ora leggere K e V non da un unico buffer contiguo per richiesta, ma **seguendo la block table**, un salto indiretto in più rispetto al kernel di attention "semplice" visto nella sezione [Attention](#attention).

Schema concettuale (semplificato, non ottimizzato — il kernel reale di vLLM è molto più elaborato per massimizzare il coalescing):

```cpp
__global__ void pagedAttentionKernel(
    __nv_bfloat16 *q,                  // query del token corrente
    __nv_bfloat16 *kv_cache_pool_k,    // pool fisico condiviso di K
    __nv_bfloat16 *kv_cache_pool_v,    // pool fisico condiviso di V
    int *block_table,                  // block_table[blocco_logico] = id pagina fisica, per QUESTA richiesta
    int seq_len,                       // lunghezza attuale della sequenza per questa richiesta
    __nv_bfloat16 *output
)
{
    // Ogni thread/warp gestisce un sottoinsieme delle posizioni della sequenza.
    // A differenza del kernel "semplice", l'indirizzo di K/V per la posizione i
    // NON è "base + i * KV_DIM" (contiguo), ma:
    int logical_block = i / PAGE_SIZE;
    int offset_in_block = i % PAGE_SIZE;
    int physical_page = block_table[logical_block];   // <-- l'indirezione in più
    __nv_bfloat16 *k_i = kv_cache_pool_k + physical_page * PAGE_SIZE * KV_DIM + offset_in_block * KV_DIM;
    __nv_bfloat16 *v_i = kv_cache_pool_v + physical_page * PAGE_SIZE * KV_DIM + offset_in_block * KV_DIM;

    // Da qui in poi, lo score di attention (q . k_i) e l'accumulo pesato con v_i
    // seguono la stessa logica del kernel di attention "semplice" —
    // cambia solo COME si calcola l'indirizzo da cui leggere K e V.
}
```

Il costo aggiuntivo rispetto al kernel "semplice" è la lettura indiretta tramite `block_table` (un ulteriore accesso a memoria per risolvere "quale pagina fisica"), che introduce un pattern di accesso meno prevedibile — un compromesso simile a quello discusso nel nostro corso a proposito del gather indicizzato per il routing MoE (Modulo 12): si scambia un po' di efficienza di accesso puro per una gestione della memoria molto più flessibile ed efficiente in aggregato. In pratica, l'implementazione reale di vLLM ottimizza pesantemente questo accesso (con letture allineate per warp, prefetching delle pagine, ecc.) proprio per minimizzare questo overhead.

---

*Fine della traduzione. Autore originale: Jędrzej Maczan, 2026, licenza Apache-2.0. Repo: [github.com/jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm). Traduzione italiana e sezioni integrative su PagedAttention a cura di Claude — vedi avvisi nel testo per distinguere le due fonti.*
