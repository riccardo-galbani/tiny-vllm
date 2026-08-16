# tiny-vllm — Corso (traduzione italiana)

> **Nota di Claude**: questa è una traduzione italiana fedele del README/corso originale di [Jędrzej Maczan](https://github.com/jmaczan/tiny-vllm) (repo `jmaczan/tiny-vllm`, licenza Apache-2.0). Ho tradotto ogni sezione controllandola riga per riga contro l'originale in inglese, per non perdere nessun esempio, digressione o nota. Il codice resta come nell'originale (variabili, commenti e nomi di funzione in inglese, come è normale per il codice sorgente). Le tre sezioni finali (PagedAttention, Paged KV cache, kernel CUDA per Paged Attention) erano, nell'originale, marcate solo con la parola "Incoming!" (l'autore non le aveva ancora scritte). Ho riportato fedelmente quel breve testo originale e poi aggiunto, **in un blocco separato e chiaramente marcato**, la mia interpretazione di cosa conterranno probabilmente, basata sul paper vLLM originale — così resta sempre distinguibile cosa è di Maczan e cosa è mio.

Stai per costruire un motore di inferenza LLM ad alte prestazioni con C++ e CUDA — tiny-vllm, un fratello minore e più piccolo di [vLLM](https://github.com/vllm-project/vllm).

Impareremo molto lungo il percorso, faremo errori e deriveremo le idee e la matematica da zero.

Questo repository è composto da due cose: 1. il codice sorgente completo del server di inferenza e 2. un corso in cui ti guido attraverso il processo di implementazione del motore. Sentiti libero di usarlo come strumento di apprendimento nel tuo percorso, oppure, se sei un docente, come risorsa didattica per la tua università.

Il motore di inferenza consiste in:

- [x] caricare un vero modello LLM da Safetensors (Llama 3.2 1B Instruct)
- [x] forward pass LLM completo (prefill + decode)
- [x] tutto il calcolo con kernel CUDA
- [x] KV cache
- [x] batching statico
- [x] batching continuo
- [x] [online softmax, in stile FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
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
- [Paged Attention](#paged-attention)
- [Paged KV cache](#paged-kv-cache)
- [Kernel CUDA per Paged Attention](#kernel-cuda-per-paged-attention)

---

## Introduzione: LLM, vLLM, modelli, server di inferenza

È facile perdersi con tutto quello che succede negli ultimi anni. Cerchiamo di fare chiarezza.

**Un LLM è un modello**. Fisicamente, **un LLM è un file che contiene tanti numeri [floating-point](https://en.wikipedia.org/wiki/Floating-point_arithmetic)**. Concettualmente, questi numeri rappresentano i pesi di alcune operazioni. I pesi vengono appresi/scoperti/trovati durante la fase di training. Alcune operazioni usano questi pesi. Ogni operazione è una funzione, che prende dei dati in input, ci fa qualcosa e produce dati in output. Le operazioni e il loro ordine sono definiti dall'architettura dell'LLM. Ogni modello ha la propria architettura, progettata da ingegneri e ricercatori.

Il processo che va da 0 a un LLM che scrive testo è così:

0. **Progettare il modello** — ingegneri e ricercatori usano un linguaggio ad alto livello come Python con una libreria per tensori come [PyTorch](https://github.com/pytorch/pytorch) o [tinygrad](https://github.com/tinygrad/tinygrad) per progettare l'architettura del modello. Addestrano versioni piccole del modello, fanno esperimenti con operazioni, dati e iperparametri (parametri delle operazioni) diversi. È la fase in cui si definisce la specifica.
1. **Implementare il modello** — Una volta decisa l'architettura finale e preparati i dati per il training, scrivono il codice che definisce il modello finale. Può essere ancora in PyTorch o simili.
2. **Addestrare il modello** — L'architettura scelta viene inizializzata con pesi fittizi. Scrivono uno script che usa ancora PyTorch o simili per far girare un algoritmo di apprendimento come la backpropagation su molto hardware, come [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) e [TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit). Questa fase brucia molta energia, denaro e potenza di calcolo. Il prodotto della fase di training è un file con i pesi del modello, in qualche formato, come il [formato Safetensors](https://huggingface.co/docs/safetensors/index). Quindi, la fase di training consiste nel trovare un insieme di pesi che produca un buon testo usando l'architettura data.
3. **Servire il modello (siamo qui)** — Il file con i pesi non può essere eseguito su un computer. Non è un eseguibile. È tanti numeri. Neanche l'architettura può essere eseguita — è solo un piano, un progetto, una descrizione del calcolo. Per far girare davvero il modello, serve un programma che trasformi l'architettura e le sue operazioni in codice eseguibile e che usi il file dei pesi per caricarli nell'architettura. Una volta scritto un programma che implementa le operazioni e che carica i pesi (i pesi vengono caricati a runtime, all'avvio del programma), puoi finalmente inviare prompt al modello e ottenere una risposta sensata. Generare un output da un modello si chiama inferenza. Ecco perché quello che costruiamo qui si chiama server di inferenza o motore di inferenza.

Conoscendo il motivo per cui serve un server di inferenza, vediamo perché lo costruiamo in C++ e CUDA. È perché vogliamo massimizzare l'uso efficiente dell'hardware e ottenere alte prestazioni. Significa che vogliamo risposte veloci ed essere in grado di gestire più prompt contemporaneamente. CUDA è l'intero ecosistema, ma anche un linguaggio che usi per scrivere codice che gira sulle GPU. Dobbiamo scrivere codice per GPU perché molte operazioni dentro un LLM consistono nel moltiplicare e sommare molti numeri. Se devi fare poca matematica, la CPU basta. Se ne devi fare tanta, la GPU è meglio. Gli LLM riguardano soprattutto la moltiplicazione di matrici, che si riduce al calcolo di prodotti scalari di due vettori, per molti numeri e per molti vettori. La matematica degli LLM è semplice, ci serviranno le basi di algebra lineare e puoi impararle mentre scrivi codice, colmando le lacune strada facendo. Trovo che questo modo di imparare "just-in-time" sia il più efficace, e forse piacerà anche a te.

La mia visione sul rapporto tra IA e calcolo, che magari trovi utile, è che **l'intelligenza deriva da tanti parametri del modello e tanto calcolo dei valori in input usando questi parametri**. Non c'è un singolo elemento a cui puoi puntare e dire: "questo è ciò che rende il modello intelligente o utile". Ogni parte del modello puoi sostituirla con una diversa e ottenere compromessi diversi, come scambiare accuratezza per complessità. Spero di non dimenticarmi di tornare su questo argomento più avanti, quando toccheremo la matematica dell'attention. Perché — il meccanismo di attention di default è molto costoso computazionalmente (O(n²·d)). E questa complessità può essere messa in discussione, e infatti alcuni lo fanno, trovando meccanismi di attention alternativi, come l'[attention lineare](https://haileyschoelkopf.github.io/blog/2024/linear-attn/). Se più persone troveranno utile questo corso, penserò a farne un altro, sui compilatori ML (uno pratico in Python o C++ più un po' di teoria SSA) o sui meccanismi di attention alternativi (matematica + kernel CUDA). Se sei interessato, fammelo sapere! Se questo corso ti risulterà utile, per favore fallo sapere ad altre persone.

> Fuori campo: la fase di training di un LLM è qualcosa che non facciamo in questo corso. Prendiamo un LLM già addestrato e scriviamo un programma che farà girare questo LLM velocemente su GPU NVIDIA per più richieste in parallelo. Se vuoi addestrare il tuo LLM, ti consiglio vivamente i repository del maestro Karpathy come [nanoGPT](https://github.com/karpathy/nanoGPT) e [llm.c](https://github.com/karpathy/llm.c) e il suo [canale YouTube](https://www.youtube.com/@AndrejKarpathy). Allo stesso modo, non progettiamo il modello, ma anche le librerie per tensori sono un argomento affascinante e vale la pena capirle da zero. [tinygrad](https://github.com/tinygrad/tinygrad) di George Hotz è un progetto che implementa una libreria per tensori con pochissimo codice, quindi se vuoi ispirarti e capire gli interni, è un buon posto per farlo (anche [il loro Discord è carino](https://discord.com/invite/ZjZadyC7PK))! C'è anche una versione un po' più vecchia e piccola di Andrej Karpathy — [micrograd](https://github.com/karpathy/micrograd). E visto che ho citato Discord, voglio consigliarti il [GPU MODE](https://discord.com/invite/gpumode) di [Mark Saroufim](https://www.marksaroufim.com/). Un sacco di persone in gamba ci girano! E se ti senti perso con quello che succede qui, e sei nuovo nel tuo percorso AI/ML, inizia con il [libro fastai](https://course.fast.ai/Resources/book.html) di [Jeremy Howard e Rachel Thomas](https://www.fast.ai/about). Ometto volutamente la parte di data science ed engineering qui, perché non ne so molto. Probabilmente [Kaggle](https://www.kaggle.com/) può essere un buon posto per iniziare e imparare facendo. Ultimo ma non meno importante, programmeremo in C++ e CUDA e useremo [cuBLAS](https://developer.nvidia.com/cublas) dove applicabile. Puoi imparare strada facendo. Le [risorse ufficiali NVIDIA](https://docs.nvidia.com/cuda/cuda-programming-guide/) sono buone e utili.

## Prerequisiti tecnici

Puoi costruirlo ed eseguirlo su qualsiasi piattaforma, con piccole modifiche, ammesso che tu abbia una GPU NVIDIA. Potresti dover aggiustare alcuni percorsi, come CUDA o GCC in [c_cpp_properties.json](https://github.com/jmaczan/tiny-vllm/blob/main/.vscode/c_cpp_properties.json) o NVCC in [CMakeLists.txt](https://github.com/jmaczan/tiny-vllm/blob/main/CMakeLists.txt).

Ti suggerisco di fare un fork di questo repo e fare gli aggiustamenti necessari perché funzioni sulla tua macchina, poi creare una pull request verso [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm) e condividere a monte le tue modifiche a beneficio di altri lettori.

Il setup esatto su cui l'autore sviluppa e testa:

- Linux (6.19.8 x64_64)
- [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (13.1)
- C++ 17
- [GCC](https://gcc.gnu.org/) (15.2.1)
- L'unica dipendenza esterna che scaricherai è il parser JSON [nlohmann/json](https://github.com/nlohmann/json) 3.12.0, che è un singolo file header — [include/json.hpp](https://github.com/jmaczan/tiny-vllm/blob/main/include/json.hpp)
- CPU AMD (Ryzen 7 9800X3D)
- GPU NVIDIA (RTX 5090)
- Usato [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) da Hugging Face (commit hash `898999bd25b40516fce5a5b8f0948f4c81c650bc`), serve solo il file `model.safetensors` da quel repository

Installa le dipendenze ed esegui il programma con `./test.sh` — costruirà ed eseguirà immediatamente.

Gira anche su GPU AMD tramite ROCm/HIP. Passa `-DUSE_HIP=ON` a CMake e compila con `hipcc` contro `hipBLAS` invece di `nvcc` e `cuBLAS`; i sorgenti CUDA vengono riusati così come sono tramite un header di compatibilità sottile `src/cuda_to_hip.h`. Scegli l'architettura della tua GPU con `-DCMAKE_HIP_ARCHITECTURES` (per esempio `gfx90a` per MI200, `gfx1100` per RDNA3, `gfx1201` per RDNA4) — non è hardcoded, quindi impostala in base alla tua scheda:

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

Un file safetensor è composto da 3 sezioni, sempre in quest'ordine: dimensione dell'header, header e dati dei tensori. La dimensione dell'header è sempre 8 byte. Questi 8 byte sono un intero senza segno a 64 bit, che dice quanti byte occupa l'header vero e proprio.

```
std::ifstream safetensors_file("model.safetensors", std::ios_base::binary);
uint64_t header_size;
safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
```

L'header è un JSON che contiene informazioni su tutti i tensori dentro il file. JSON è solo un gruppo di coppie <chiave, valore>, dove la chiave è una stringa unica con il nome del tensore e il valore è un altro oggetto JSON, con informazioni su quel tensore. Ogni chiave in questo JSON è il nome di un tensore, tranne un'unica chiave chiamata `__metadata__`, probabilmente per informazioni aggiuntive quando necessario (non la useremo, le specifiche dicono che è una "chiave speciale per memorizzare una mappa testo-a-testo in forma libera"). Ogni valore è un JSON contenente tre chiavi — `dtype`, `shape` e `offsets`. `dtype` dice in che tipo di dato è memorizzato il tensore. `shape` dice le [dimensioni del tensore](https://en.wikipedia.org/wiki/Tensor#As_multidimensional_arrays) e `offsets` dice dove è memorizzato il tensore, all'interno della sezione dati dei tensori. Ogni `shape` è una lista di interi di lunghezza sconosciuta, e ogni valore `offsets` è un vettore di esattamente due interi. Il primo elemento dice dove inizia il tensore e l'ultimo dice dove finisce.

A questo punto affronti la prima decisione di design. Vuoi rendere l'architettura del tuo server indipendente dal modello, così può far girare qualsiasi modello arbitrario, purché tu implementi le operazioni di cui ha bisogno, oppure vuoi iniziare in modo semplice concentrandoti sul modello scelto?

Qualunque cosa tu decida, è sempre più facile sviluppare su un singolo modello e poi generalizzare, piuttosto che cercare di renderlo flessibile fin dall'inizio quando non sei sicuro di come apparirà il codice alla fine. Puoi sempre tornarci sopra e aggiornarlo quando scegli di farlo.

Se vuoi rendere il tuo server indipendente dal modello, devi allocare la memoria, impostare `shape` e tipo (`dtype`) dei dati del modello dinamicamente in base all'header Safetensors e implementare più operazioni, assicurandoti di coprire tutte le operazioni usate da tutti i modelli che vuoi supportare. Probabilmente dovresti comunque fornire dei progetti dell'architettura dei modelli, perché il file Safetensors non ti dice quale operazione eseguire con quei dati, in che ordine, ecc. L'autore non è sicuro di quale sia l'approccio ottimale, ma se lo scopri, da solo o leggendo il codice di vLLM/[TensorRT](https://github.com/NVIDIA/TensorRT)/altri, condividi pure le tue scoperte.

L'autore assume di codificare il server solo per l'architettura di Llama 3.2 1B Instruct. Ecco un dump dell'oggetto [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) `LlamaForCausalLM` con il modello `meta-llama/Llama-3.2-1B-Instruct` caricato. Possiamo ispezionare quali operazioni dobbiamo implementare e quale forma e tipo di dati dobbiamo usare:

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

Non preoccuparti se alcune di queste operazioni non ti sono ancora familiari. Le capirai in modo viscerale man mano che procediamo nel corso. Anche l'autore continua a dimenticare come funzionano, a volte, e deve spesso ricontrollare, quindi non sentirti male se vai avanti e indietro o cerchi informazioni con un chatbot o una ricerca su internet.

## Come funzionano i numeri floating-point e perché usiamo bfloat16

Ricordiamoci cosa vogliamo ottenere. Vogliamo caricare un modello. Conosciamo già la struttura del file con un modello, un file Safetensors. Conosciamo l'architettura del nostro modello di riferimento. Abbiamo verificato che i pesi del modello sono memorizzati in tipo BF16. Prendiamoci un momento per pensare a questo tipo e ai numeri floating-point in generale.

Tutto sul computer è binario, alla fine. [Anche i formati numerici dei computer](https://en.wikipedia.org/wiki/Computer_number_format) lo sono.

Nei modelli ML, i pesi quasi mai sono numeri interi. Sono piuttosto numeri reali. E i computer sono binari. Significa che bisognava capire come rappresentare numeri reali nei linguaggi di programmazione, in un modo che sia efficiente in termini di memoria. In questo contesto, significa che puoi impacchettare molta informazione in poco spazio. Perché, certo, potresti costruire un tipo di dato complesso ("complesso" nel senso di "non banale", non nel senso di "reale+immaginario"), dove la parte a sinistra del separatore decimale è memorizzata come intero e la parte dopo il separatore (quindi la parte minore di 1) è memorizzata come un altro intero. Ma puoi vedere quanto sia inefficiente. È utile solo per situazioni in cui hai davvero bisogno della piena precisione. Ed è per questo che esistono cose come [decimal in Python](https://docs.python.org/3/library/decimal.html).

Ok, allora pensiamo a cosa potremmo fare invece. L'alternativa più semplice sarebbe prendere un numero intero, tipo 1234, e dire "metterò un punto tra 12 e 34", ottenendo così 12.34. Le persone hanno risolto questo con un fattore di scala: letteralmente, per 1234 useresti un fattore di scala 1/100 per ottenere 12.34. Devi comunque capire come rappresentarlo sul computer. In questo formato, la parte frazionaria (dopo il punto) ha sempre lunghezza fissa — è esattamente nel mezzo. Questo formato si chiama [numeri a virgola fissa](https://en.wikipedia.org/wiki/Fixed-point_arithmetic) (fixed-point). È meno diffuso dei numeri a virgola mobile (floating-point).

Quindi, se esistono i numeri a virgola fissa, devono inevitabilmente esistere anche numeri non a virgola fissa. Pensiamoci, in modo approssimativo. "Non a virgola fissa" potrebbe significare che il punto (la virgola) può muoversi. La parte frazionaria può essere più lunga o più corta. Sembra una soluzione più efficiente in termini di memoria.

I numeri floating-point, come tutti gli altri tipi numerici, sono rappresentati come sequenze di bit. Hanno tutti scelte di design leggermente diverse, quindi concentriamoci sul normale [float a 16 bit](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16) (float16, FP16, IEEE 754-2008).

Float16 e molti altri numeri floating-point funzionano in modo simile. Consistono in tre sezioni: segno, esponente e frazione. In totale occupano 16 bit.

```
[ sign | exponent  | fraction            ]
[ 0    | 0 1 0 0 1 | 0 0 1 0 0 0 0 0 0 0 ]
```

L'intuizione è questa: nei numeri floating-point puoi spostare il punto. La frazione è il numero che usi per spostare il punto. Come nell'esempio precedente, la tua frazione può essere 1234. Stavolta, il punto non è fisso, quindi può stare ovunque tu scelga — 1.234, 0.0001234, 1234000, ecc.

Il segno è un singolo bit. 0 significa che il numero è positivo e 1 significa negativo.

L'esponente è 5 bit. Controlla la grandezza del numero.

La frazione è 10 bit. Controlla quale numero usiamo per spostare il punto. Nomi alternativi: significand, mantissa.

Ora la formula: $(-1)^{segno} \times 2^{esponente-bias} \times (1.frazione)$

Nota due cose nuove — l' $1.$ prima della frazione e il $bias$. L'`1.` è una scelta di design per aumentare la precisione del numero (guadagni 1 bit gratis) senza memorizzarlo esplicitamente in memoria. Ecco perché si chiama implicito. Vedi [questa bella domanda e risposta su Stack Overflow](https://stackoverflow.com/questions/4930269/floating-point-the-leading-1-is-implicit-in-the-significand-huh) per una spiegazione migliore. L'idea principale è che non è qualcosa di obbligatorio, ma piuttosto un trucco intelligente per memorizzare più dati senza usare più memoria, dicendo nella specifica dei numeri floating-point che quel bit c'è sempre (tranne per lo zero). La prossima cosa nuova è il $bias$ — guarda di nuovo dove si trova: $2^{esponente-bias}$. Hai qualche sospetto sul perché sia lì?

La risposta è che serve a poter rappresentare anche numeri più piccoli di 1. Nello specifico, lo fa rendendo possibile avere una potenza negativa di $2$. L'esponente è sempre un numero positivo, rappresentato con pochi bit (5 bit, in float16). Se non sottraessimo il bias da esso, $2^{esponente}$ sarebbe sempre positivo, e probabilmente un numero piuttosto grande (11111 binario è 31 decimale). Quindi potremmo generare qualsiasi numero maggiore di 1 o minore di -1 (quasi qualsiasi numero, limitato dalla precisione del numero floating-point), ma non potremmo rappresentare nessun numero piccolo ma maggiore di 0, come 0.0001234. Da qui, dobbiamo sottrarre il bias dall'esponente. Il bias non può essere né troppo grande né troppo piccolo. Sapendo questo, quale pensi sia il bias per float16? Ricorda che il valore massimo dell'esponente è 31.

Quindi il bias per float16 è 15. Significa che puoi rappresentare sia numeri più piccoli che più grandi.

Ci sono altre cose interessanti nella specifica dei numeri floating-point, come codifichiamo gli infiniti positivi e negativi, come codifichiamo il valore non-un-numero (NaN), ecc. [Wikipedia ha molti esempi sulla codifica dell'esponente](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Exponent_encoding).

> Le dimensioni di esponente e frazione sono le principali differenze tra i diversi tipi di numeri floating-point.

Prendiamo un esempio e codifichiamo il numero 12.34 come float16. 12.34 è positivo, quindi il bit di segno è 0. Per rappresentare 12.34, dobbiamo trasformare 12 in binario:

- 12 / 2 = 6, resto 0
- 6 / 2 = 3, resto 0
- 3 / 2 = 1, resto 1
- 1 / 2 = 0, resto 1

Leggilo al contrario e ottieni 1100.

Quindi il decimale 12 è il binario 1100.

Ora trasformiamo la frazione 0.34 in binario:

- 0.34 * 2 = 0.68, parte intera è 0
- 0.68 * 2 = 1.36, parte intera è 1, sottraiamo 1 e continuiamo
- 0.36 * 2 = 0.72, parte intera è 0
- 0.72 * 2 = 1.44, parte intera è 1, sottraiamo 1 e continuiamo
- 0.44 * 2 = 0.88, parte intera è 0
- 0.88 * 2 = 1.76, parte intera è 1, sottraiamo 1 e continuiamo
- 0.76 * 2 = 1.52, parte intera è 1, sottraiamo 1 e continuiamo
- 0.52 * 2 = 1.04, parte intera è 1, sottraiamo 1 e continuiamo
- 0.04 * 2 = 0.08, parte intera è 0
- 0.08 * 2 = 0.16, parte intera è 0

A questo punto, abbiamo calcolato 10 bit, che è la nostra lunghezza di frazione/mantissa/significand. A prima vista, sembra che abbiamo calcolato esattamente tanti bit quanti ce ne servono, ma non è del tutto corretto. Perché dobbiamo includere nella mantissa anche la prima parte del numero che stiamo codificando — 12 (1100 in binario). L' $1.$ è sempre una parte implicita del numero, quindi da $1100$ rimuoviamo il primo $1$. Dobbiamo usare quello che resta — $100$ — dentro la mantissa. Significa che abbiamo calcolato 3 bit in eccesso, che scarteremo, a causa dei 3 bit che dobbiamo usare per includere $12$ nella mantissa. Quindi non abbiamo fatto un errore, ma abbiamo fatto un po' di lavoro non necessario. Vale la pena ricordarlo se decidi mai di implementare il tuo tipo numerico (e se lo fai da solo, [fammelo sapere](mailto:jedrzej@maczan.pl)!).

A proposito, guarda l'ultimo termine: "0.08 * 2 = 0.16, parte intera è 0". Il processo non è finito, perché non abbiamo raggiunto 0 (ci siamo fermati a 0.16). Significa che perderemo precisione nel rappresentare questo numero.

Leggendo i bit dall'alto verso il basso, il decimale 0.34 è approssimativamente il binario 0101011100.

Mettiamo insieme 12 e 0.34 e otteniamo 1100 e 0101011100. Di nuovo, $1.0$ è implicito, quindi ci restano 100 e 0101011100. Mettili insieme e hai 1000101011100. Sono 13 bit. Rimuovi i 3 bit meno significativi (quelli più a destra) e la rappresentazione binaria finale della frazione del nostro numero è 1000101011.

> So che sto divagando un po', dovevi costruire un server di inferenza. Ma onestamente, perché dovremmo avere fretta? Se queste cose non ti interessano, puoi sempre saltare avanti. Credo che ci sia una gioia intrinseca nell'imparare le cose in profondità e se sei d'accordo con me su questo, continuiamo.

Abbiamo finito con la mantissa. Guarda come abbiamo trasformato $1100$ in $1.100$. Se fossimo ancora nel mondo decimale (base-10), diremmo che per trasformare $1.100$ in $1100$ dobbiamo moltiplicarlo per $1000$ ($10^3$). I numeri con cui lavoriamo sono binari (base-2), quindi significa che dobbiamo moltiplicarlo per $2^3$, il che ci ricorda immediatamente $2^{esponente}$ dalla formula del floating-point. Il che significa che il nostro esponente è $3$. Ma, il bias. Il bias è 15. E nella formula per decodificare il float, nell'esponente di $2$ sottraiamo il bias dall'esponente che leggiamo dai bit del floating-point (questi 5 bit). Significa che dobbiamo aggiungere il bias così che compaia nell'esponente memorizzato. Quindi sommiamo $3+15$ e il nostro esponente finale è $18$. Trasformiamo velocemente 18 in formato binario:

- 18 / 2 = 9, resto 0
- 9 / 2 = 4, resto 1
- 4 / 2 = 2, resto 0
- 2 / 2 = 1, resto 0
- 1 / 2 = 0, resto 1

Leggi al contrario e il decimale 18 è 10010 binario.

Possiamo mettere tutto insieme per vedere finalmente il decimale 12.34 come float16:

- bit di segno: 0 (numero positivo)
- esponente: 10010 (18 binario, 3 perché abbiamo spostato a sinistra di 3 posizioni + 15 di bias; totale 5 bit)
- frazione: 1000101011 (totale 10 bit)

Il numero di bit somma correttamente a 16. Insieme appare così:

```
[ sign | exponent  | fraction            ]
[ 0    | 1 0 0 1 0 | 1 0 0 0 1 0 1 0 1 1 ]
```

Una cosa interessante è che sappiamo di aver perso un po' di precisione trasformando 12.34 in un float. È successo quando abbiamo calcolato la parte 0.34. Che ne dici di verificare da soli e decodificare il nostro float di nuovo in decimale, per vedere quale valore memorizza davvero il computer quando proviamo a memorizzare 12.34 come float16? Sei curioso anche tu?

Ricorda la formula: $(-1)^{segno} \times 2^{esponente-bias} \times (1.frazione)$

Sappiamo che il numero è positivo, perché il segno è 0, quindi $-1^0 = 1$.

Sappiamo che l'esponente è 18 meno il bias 15, quindi è 3. Inseriamolo nella formula e abbiamo: $2^3=8$.

La frazione è 1000101011 e dobbiamo trasformarla in decimale. Dobbiamo ricordarci anche dell'$1.$ implicito, quindi tutto insieme è $1.1000101011$. Ogni cifra successiva viene moltiplicata per una potenza di 2, ma con un esponente minore di 1 rispetto alla precedente. Simile a come funziona quando trasformiamo un binario in decimale — facciamo lo stesso ma al contrario, partendo da destra moltiplichiamo ogni cifra per 2 elevato alla potenza della posizione corrente. Verifichiamolo davvero: l'esponente 10010 è quindi $0 \times 1 + 1 \times 2 + 0 \times 4 + 0 \times 8 + 1 \times 16 = 18$, proprio come pensavamo. Ok, ora la mantissa. Trasformiamo $1000101011$ in decimale: $1 \times 2^{-1} + 0 \times 2^{-2} + 0 \times 2^{-3} + 0 \times 2^{-4} + 1 \times 2^{-5} + 0 \times 2^{-6} + 1 \times 2^{-7} + 0 \times 2^{-8} + 1 \times 2^{-9} + 1 \times 2^{-10} = 0.5 + 0 + 0 + 0 + 0.03125 + 0 + 0.0078125 + 0 + 0.001953125 + 0.0009765625 = 0.5419921875$. Quindi, mettendolo insieme con l'$1.$ implicito, il nostro numero è $1.5419921875$.

Inseriamo tutto insieme nella formula: $1 \times 8 \times 1.5419921875 = 12.3359375$. Corretto e abbastanza vicino, almeno per l'inferenza LLM. L'errore: $12.34 - 12.3359375 = 0.0040625$.

Ma questo era float16. E il nostro modello di riferimento (Llama 3.2 1B Instruct) usa invece bfloat16 (BF16). Si scopre che bfloat16 è ampiamente usato in molti modelli diversi. Occupa anch'esso 16 bit, ma il suo esponente è più lungo (8 bit) del normale float16 (5 bit). La prima cosa interessante di bfloat16 è che la dimensione dell'esponente è in realtà la stessa di un float a 32 bit, il doppio più grande (float32, FP32, float), che ha anch'esso un esponente a 8 bit. Quindi, confrontandolo con float16, bfloat16 scambia 3 bit di frazione per guadagnare 3 bit di esponente. Naturalmente, la frazione si riduce, ed è ora solo 7 bit. Riassumendo — bfloat16 è 1 bit di segno, 8 bit di esponente e 7 bit di frazione.

Perché conta bfloat16? Ha la stessa dimensione di float16 (metà di un float a piena precisione), e allo stesso tempo ha lo stesso esponente di float32 — al costo di una frazione più piccola. L'industria spesso lo sceglie per l'inferenza, perché è meno soggetto a problemi di range ([overflow](https://en.wikipedia.org/wiki/Integer_overflow) / [underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow)) e allo stesso tempo la perdita di precisione (dovuta alla frazione più piccola) è un compromesso accettabile nell'inferenza LLM (i risultati empirici lo dimostrano).

Se vuoi mettere alla prova la tua comprensione, la pagina Wikipedia sui formati floating-point a mezza precisione ha [tanti buoni esempi](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples).

Ora hai tutti i prerequisiti per caricare un modello nel tuo motore di inferenza e iniziare a fare cose interessanti. Vorresti leggere un'introduzione pratica su come lavorare con la memoria GPU in CUDA? Continua a leggere la prossima sezione, oppure salta direttamente alla parte di inferenza.

Ah, a proposito, i float a 32 bit si chiamano float a precisione singola. E poi senti parlare di doppia precisione. E potresti pensare — quindi ottengo DUE punti mobili nello stesso numero? Purtroppo la doppia precisione significa solo che il numero è più grande (64 bit). Bah.

## Memoria GPU e CPU

Questo capitolo potrebbe sembrare un po' casuale, ma spero sia utile se sei ancora agli inizi con la programmazione GPU. I dati possono vivere sull'host o sul device.

L'host è il tuo PC con la CPU. Ha memoria lenta e grande — DRAM — quella che compri e inserisci nella scheda madre. Recentemente molto costosa, a causa della frenesia AI del big tech. La DRAM è separata dalla CPU. La CPU ha anche una propria memoria on-chip piccola e veloce — SRAM.

Il device è la tua GPU. Ha anch'essa memoria lenta e grande — HBM, VRAM — ma è saldata sulla scheda e non la puoi espandere facilmente come fai con la DRAM nel tuo PC. Ha memoria piccola e veloce — SRAM — che puoi usare se definisci variabili `__shared__`.

La GPU non può accedere alla tua DRAM, quindi prima di eseguire qualsiasi calcolo sulla GPU, devi copiarci i dati. Il flusso tipico può essere così:

1. Crea una variabile sulla CPU
2. Scrivi sulla variabile sulla CPU
3. Calcola la dimensione di memoria occupata dalla variabile sulla CPU, oppure trova la dimensione massima in altro modo (a volte conosci già il numero massimo di elementi in un vettore o simili)
4. Moltiplica la dimensione calcolata per la dimensione del tipo di variabile
5. Alloca la memoria sulla GPU
6. Copia la variabile dalla CPU alla GPU
7. Ora puoi usare questi dati nei tuoi calcoli GPU

Idealmente vuoi allocare meno memoria possibile, riusare questa memoria il più possibile e copiare i dati il meno possibile.

Facciamo un esempio: devo sapere quali token sono attualmente attivi durante l'inferenza. Per questo scopo, mi servono questi dati sia su CPU che su GPU.

1. Creo un vettore sulla CPU:

```
std::vector<int> active_tokens;
```

2. Scrivo dei token nel vettore, tipo

```
active_tokens.push_back(token);
```

3. Ora devo capire quanti `active_tokens` potrebbero mai esserci (il numero massimo), per allocare la memoria GPU di conseguenza. In questo caso, so che potrebbero essere fino a `BATCH_SIZE` (diciamo, massimo 2 elementi sempre)
4. I token sono interi, quindi la dimensione finale di memoria da allocare è

```
BATCH_SIZE * sizeof(int)
```

5. Dichiaro e alloco la memoria GPU

```
int *gpu_active_tokens;
cudaMalloc(&gpu_active_tokens, BATCH_SIZE * sizeof(int));
```

`cudaMalloc` è la funzione CUDA che usi per allocare memoria GPU. Ti serve un puntatore sull'host — `int *gpu_active_tokens` — per poter usare questa memoria e passarla ai tuoi kernel GPU.

6. Copio i dati dalla CPU alla GPU. Conosciamo la quantità massima di token attivi (`BATCH_SIZE`), ma nel caso in cui attualmente ci siano meno token attivi processati, dobbiamo calcolarne la quantità: `num_active_slots * sizeof(int)`

```
//               destinazione,               sorgente,           dimensione dati da copiare,   direzione della copia
cudaMemcpy(gpu_active_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
```

7. Ora puoi usare questi dati quando invochi i kernel CUDA, tipo

```
embeddingGatherDecode(gpu_active_tokens, num_active_slots, hidden_state, weights.embed_tokens);
```

## Inferenza di un singolo token

Iniziamo a lavorare sull'inferenza ora. Puoi chiudere questo corso ora e iniziare a scrivere codice, basandoti sulla sequenza di operazioni delineata nella sezione [Safetensors e il tuo modello](#safetensors-e-il-tuo-modello), oppure puoi continuare a leggere per riempirti la cache del cervello di cose utili grazie alle quali ti sarà più facile implementare il modello. Scegli tu.

Indipendentemente da cosa scegli, vorrei condividere con te com'è imparare cose nuove usando gli LLM al 2026, magari lo troverai utile mentre lavori a questo corso. L'autore pensa al processo di apprendimento come a questo ciclo:

1. Capisci più o meno cosa vuoi costruire
2. Descrivi il tuo modello mentale a un chatbot, chiedigli di individuare i tuoi punti ciechi, errori nel tuo ragionamento, colmare le lacune nella tua conoscenza con informazioni su misura e abbastanza contesto per capire
3. Leggi la risposta, prova a interiorizzarla e aggiorna il tuo modello mentale
4. Ripeti finché non ci sono più errori
5. Inizia a scrivere codice, continua finché non sei bloccato
6. Quando sei bloccato, torna al punto 2

Una cosa che l'autore trova molto importante è ricordare che un essere umano impara mettendo un vero sforzo nel capire. Capire le cose, fare errori, fare il debug dei propri processi di pensiero, pensare al proprio pensiero, aggiornare la propria comprensione delle cose descrivendole ad altri e riconoscere i punti in cui la propria comprensione è insufficiente o sbagliata. Tutte queste cose vanno fatte intenzionalmente, perché potresti sempre delegarle a un LLM. L'autore è un forte sostenitore degli LLM come tutor personalizzato, ed è la cosa migliore con cui imparare, al 2026. Inoltre, pensa che ci siano 2 tipi diversi di sforzo, ed è da lì che nasce la confusione su se e come imparare con gli LLM. Un tipo di sforzo è lo sforzo di comprensione, dove ricostruisci il tuo cervello pensando a una cosa che vuoi capire. La scomponi in pezzi più piccoli, capisci il meccanismo, il flusso dei dati, le relazioni, la colleghi ad altre cose che sai, immagini, ricrei, fai errori e questi ti indicano dove guardare dopo. Scavare in profondità in un concetto che vuoi imparare è una delle più alte bellezze della vita. Questo tipo di sforzo e fatica è dove avviene l'apprendimento. E poi c'è anche uno sforzo che spreca energia. Cose come — recupero di dati, spostamenti, azioni meccaniche che consumano il tuo tempo, che sei perfettamente in grado di fare da solo, le capisci dall'interno all'esterno, e non ti insegnano più nulla di nuovo — questo è un tipo di sforzo che l'autore pensa abbia un ROI troppo basso da eseguire, ed è meglio delegarlo a un LLM. Puoi obiettare che in un certo senso stai comunque delegando il tuo pensiero a un LLM, e non avresti torto. Puoi sempre migliorare nella lettura della documentazione (lo farai comunque mentre impari, ma spesso puoi saltare questa parte per cose una tantum) o nella ricerca su internet. Ma il tuo tempo, la tua attenzione e la tua energia sono limitati. Se diventare marginalmente migliore nella ricerca su internet avviene a spese dell'imparare di più sull'argomento che stai studiando in questo momento — allora non è un buon compromesso. Le tue esperienze potrebbero variare, comunque, l'autore non è un esperto, è solo una persona a caso su internet.

Hai la comodità di poter sempre dare un'occhiata al codice dell'autore se vuoi capire come l'ha costruito lui. La maggior parte delle volte, probabilmente non ne hai bisogno. Ma se sei bloccato o quando finisci una certa cosa e vuoi confrontare — allora ovviamente sei incoraggiato a farlo.

L'autore ti guiderà attraverso il progetto di un programma — da una pagina vuota all'inferenza di un singolo token. Ora hai molte informazioni di cui avevi bisogno per iniziare.

Quindi, devi fare il bootstrap del progetto. Importa il runtime CUDA, cuBLAS e `nlohmann::json`. Questo è il tipo di lavoro che puoi liberamente copiare-incollare dal codice dell'autore, se non vuoi farlo da solo.

```
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0 // questo è utile, ma l'autore non ricorda perché
#include "json.hpp"

using json = nlohmann::json;
```

Se usi VS Code e si lamenta di import non risolti, devi aggiustare i percorsi nei file di `.vscode/`.

Una funzione helper utile che potresti eseguire una volta all'inizio della tua funzione `main()` sono alcune informazioni di debug sulla GPU che CUDA ha scelto:

```
int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}
```

Ok, ora carica il modello. Conosci già la struttura del file safetensors. Ci sono molti approcci possibili su come memorizzerai il modello. L'approccio che piace all'autore, e che pensa possa piacere anche a te, è caricare i tensori come un unico grande blocco nella memoria GPU. Devi prima allocare questa memoria e l'header del file safetensors ti dirà quanta. Poi, usando gli offset nell'header, mapperai le regioni nella memoria GPU su puntatori sulla CPU, così sai dove guardare quando vuoi, ad esempio, recuperare i pesi di K nel layer 5. Alcuni pesi non sono specifici di un layer, assicurati di riconoscerlo. Un consiglio: fatti un favore e stampa molto. I debugger sono ottimi, ma non così utili per lavorare con dati grezzi. Scrivi (o fai generare da un LLM) degli script helper in Python per stampare i dump dei modelli. L'autore fa molte cose di questo tipo e lo aiutano a capire i dati, il modello, ecc. Qualsiasi cosa ti aiuti ad andare avanti e rafforzi la tua comprensione va bene, secondo l'autore.

Ah, a proposito, il tipo di dato del modello è bfloat. In CUDA, è `__nv_bfloat16`.

Se implementi i pesi in modo simile a quanto descritto sopra, allora per recuperare i pesi di K in un layer, scriverai qualcosa del genere:

```
weights.w_k[5] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers.5.self_attn.k_proj.weight"));
```

Devi allocare dati per tutti i tensori che il tuo modello usa. Non aver paura di allocare troppo e renderlo non ottimale. Il primo obiettivo è farlo funzionare.

Nella sezione [Safetensors e il tuo modello](#safetensors-e-il-tuo-modello) abbiamo delineato la sequenza esatta di operazioni che dobbiamo implementare per predire il primo token. Una volta che lo facciamo funzionare e otteniamo davvero il nostro primo token generato, riuseremo molto del codice e scriveremo un loop attorno ad esso.

## Tokenizzazione

Ok, quindi l'autore assume che tu abbia caricato il modello e mappato i pesi del modello su alcuni puntatori utili. Ora dobbiamo leggere l'input dell'utente, il prompt, e trasformarlo da testo in qualcosa che il modello capisce — i token. Tutti gli LLM mainstream usano token, non parole o caratteri.

Per trasformare il testo in una sequenza di token, serve un tokenizer. Useremo un tokenizer esistente, che produce token corrispondenti al dizionario di Llama 3.2 1B. Vedi il file [python/tokenizer.py](https://github.com/jmaczan/tiny-vllm/blob/main/python/tokenizer.py), dove l'autore usa un tokenizer di Hugging Face.

Andare a fondo sui tokenizer è fuori dallo scopo di questo corso; quello che devi davvero ricordare è che prende un testo e produce una sequenza di token (interi), che rappresentano il tuo testo come vettore di interi. E l'LLM ha bisogno del tuo testo come questo vettore di interi.

> Costruire il proprio tokenizer è una cosa piuttosto divertente. L'autore ha scritto il suo 3 anni fa e ti invita a usarlo come riferimento, se vuoi saperne di più sui tokenizer: <https://github.com/jmaczan/bpe-tokenizer>. C'è anche un'ottima risorsa di Andrej Karpathy dove costruisce un tokenizer, ed è molto utile ed educativa: video <https://www.youtube.com/watch?v=zduSFxRajkE>, codice <https://github.com/karpathy/minbpe> e questo articolo <https://github.com/karpathy/minbpe/blob/master/lecture.md>.

## Embedding

Il tuo testo è tradotto in token e li dai in pasto al tuo server di inferenza LLM. I token sono più che altro indici, ma non sono i dati su cui lavorerà davvero il tuo LLM. I modelli linguistici di grandi dimensioni sanno come mappare ogni token a un vettore, dove ogni token ha la stessa lunghezza di vettore, ma valori diversi. Questi vettori si chiamano embedding. Incorporano il significato di un token. Poi dai in pasto una lista di token all'LLM, che recupera un embedding per token, dove i token funzionano come indici che dicono al modello quale embedding (vettore) recuperare dai suoi pesi. Nel nostro caso, ogni embedding ha lunghezza 2048. Quindi per 5 token in input, ottieni 5 vettori di lunghezza 2048, che insieme formano una matrice di dimensione (5, 2048). Conosciamo già il tipo di ogni numero in questi vettori di embedding — è bfloat16.

Questo potrebbe essere il primo kernel CUDA da scrivere in questo corso. Il tuo compito è recuperare gli embedding per tutti i token in input.

Di cosa hai bisogno per farlo? Hai bisogno dei token in input e dei pesi degli embedding. Hai già caricato i pesi degli embedding sulla tua GPU. Ma i token in input che fornisci vivono sulla CPU — puoi fornirli come parametri da riga di comando, hardcoderli o leggerli da un file. Di default, probabilmente li carichi in qualche vettore di interi. Ora devi renderli disponibili alla tua GPU. Quindi cosa fai ora?

Puoi creare un buffer sulla GPU in cui copi i tuoi token in input. Quando lo fai, sarai in grado di usare il puntatore a questi token in input sulla GPU e passare il puntatore ai tuoi kernel CUDA. Questa volta, l'autore ti aiuta a farlo. Qualsiasi altro kernel e spostamento/allocazione dati CUDA lo scriverai da solo, a meno che non siano eccezionalmente interessanti, ok?

Diciamo che hai i tuoi token in input in un vettore di interi sulla CPU:

```
std::vector<int> input_tokens = {678, 264, 1933, 13};
```

Il tuo modello, come tutti i modelli, ha un vincolo su quanti token può processare. Per Llama 3.2 1B sono 2048. Include sia i token in input che quelli in output. Dobbiamo scegliere arbitrariamente quanti di essi permettiamo che siano la dimensione del prompt. Diciamo che saranno massimo 512 token come input e lo dichiariamo come [constexpr](https://en.cppreference.com/w/cpp/language/constexpr.html):

```
constexpr int MAX_PROMPT_LEN = 512;
```

Ti serve una copia dei tuoi token in input sulla GPU. Quindi, devi allocare memoria sulla GPU e ti serve un puntatore a questa memoria. Un puntatore:

```
int *gpu_input_tokens;
```

Per allocare la memoria sulla GPU, useremo la funzione `cudaMalloc`, che già conosci dal capitolo sulla memoria GPU. Il primo argomento di `cudaMalloc` è un puntatore al nostro puntatore (`void **`). Il secondo argomento è la dimensione di memoria da allocare. Sappiamo qual è il numero massimo di token che possiamo avere in input. I token sono interi, quindi la dimensione di memoria finale da allocare è il numero massimo di token in input * dimensione di un intero.

```
cudaMalloc(&gpu_input_tokens, MAX_PROMPT_LEN * sizeof(int));
```

Pronti a copiare i token in input nella GPU ora:

```
//              destinazione,              sorgente,                              dimensione, direzione della copia
cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
```

Possiamo scrivere un kernel CUDA ora.

## Ingegneria di kernel CUDA — embedding

> Esistono risorse molto migliori di quanto l'autore possa produrre, quindi per imparare CUDA, consulta la [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html). L'autore accenna solo brevemente le basi qui, ma potrebbero non essere sufficienti per te.

I kernel sono funzioni eseguite su una GPU. Lanci la stessa funzione più volte. Ogni funzione lanciata è un thread separato. Eseguono lo stesso codice. Ricevono parametri leggermente diversi, come l'indice di un thread. I thread sono raggruppati in block. Quando lanci un kernel CUDA, definisci quanti block vuoi invocare e quanti thread ci sono in ogni block. I thread sono anche raggruppati in warp. Ogni warp ha 32 thread. Quindi, se lanci il tuo kernel e definisci che deve eseguire 5 block e ogni block deve eseguire 64 thread, allora significa che ogni block esegue 2 warp, 32 thread ciascuno.

Quando si scrivono kernel CUDA, molto sforzo va nel pensare alla memoria. Cioè, è come pensare dalla prospettiva del thread: "quali dati dovrei processare?", "dove dovrei scrivere i risultati?". In pratica significa capire l'indice dei dati in input da leggere e dove scrivere l'output. I tuoi strumenti principali sono variabili built-in, come `threadIdx`, `blockIdx` e `blockDim` — vedi [questo link per informazioni più strutturate](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html). Ogni thread ha i propri valori per queste variabili. Questo rende possibile eseguire lo stesso calcolo in parallelo. Questo approccio si chiama [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads).

Ok, quindi sappiamo che per ogni token in input, vogliamo recuperare un embedding composto da 2048 numeri bfloat16. Il primo approccio a cui possiamo pensare: okay, abbiamo N token da recuperare, e ogni token deve recuperare 2048 numeri. Quindi possiamo lanciare N block — un block per token — e 2048 thread in ogni block, così ogni thread recupererebbe esattamente un numero. Scriviamo un kernel vuoto e annotiamo come vorremmo eseguirlo con N block e 2048 thread per block.

Kernel vuoto:

```
__global__ void embeddingGatherKernel()
{
}
```

Un'invocazione di questo kernel:

```
embeddingGatherKernel<<<num_input_tokens, 2048>>>();
```

Sembra a posto. Ora pensiamo a che dati ci servono in input e cosa vogliamo produrre. Ci servono i token in input e i pesi degli embedding del modello caricato. Ci serve anche un output su cui scrivere. Abbiamo già copiato i token sulla GPU e abbiamo i pesi degli embedding sulla GPU. Quindi l'output, com'è fatto? Recuperiamo 2048 numeri bfloat16 per N token. Quindi ci serve una memoria, su cui possiamo scrivere quel numero di numeri. Allochiamola prima:

```
__nv_bfloat16* input_embeddings;
cudaMalloc(&input_embeddings, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * 2048);
```

Dimensione della memoria che allochiamo: numero massimo di token in input (MAX_PROMPT_LEN) * 2048 * dimensione di ogni numero (sizeof(__nv_bfloat16)).

Passiamo quindi questi puntatori nel nostro kernel. `weights.embed_tokens` sono i pesi degli embedding che abbiamo caricato dal file safetensors:

```
embeddingGatherKernel<<<num_input_tokens, 2048>>>(gpu_input_tokens, input_embeddings, weights.embed_tokens);
```

Aggiorniamo la firma del kernel per accettare questi argomenti:

```
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
```

Okay. Ora dobbiamo effettivamente recuperare i numeri. Ricorda, pensiamo al kernel dalla prospettiva di un singolo thread, dove ci sono più thread invocati con esattamente lo stesso codice, ma i loro valori di `threadIdx.x`, `blockIdx.x` (e alcune variabili simili) sono diversi — è così che sanno chi sono e quali dati devono prendere e quali dati devono produrre.

Calcoliamo prima l'indice dell'output. Sappiamo che è di dimensione (numero di token, 2048). Assumiamo che sia in formato row-major, così è più facile lavorarci. In altre parole, significa che quando accedi all'indice 0, ottieni il numero 0 del token 0, indice 1, ottieni il numero 1 del token 0, e così via fino all'indice 2047. Poi, ricomincia da capo con il numero 0 del token 1. `threadIdx.x` è unico all'interno di un block. Quindi, se avessimo un solo token, potremmo mappare 1:1 `threadIdx.x` all'indice di `input_embeddings` (la memoria di output), tipo `input_embeddings[threadIdx.x]`. Per un momento, assumiamo che sia questo il caso — che riceviamo sempre un singolo token. Quindi, ora sappiamo dove scrivere. Dobbiamo recuperare il numero corretto di un embedding per questo token. Ogni embedding è di 2048 numeri, quindi per arrivare all'inizio dell'embedding dell'id del token, dobbiamo moltiplicare il token per 2048 — `gpu_input_tokens[0] * 2048` (di nuovo, assumendo che `gpu_input_tokens` sia un singolo token, quindi possiamo accedere al suo primo elemento). In questo modo riceviamo l'indice del primo numero dell'embedding del token. L'embedding ha 2048 numeri e ogni thread in un block recupera un singolo numero di questo embedding. Quindi, dobbiamo spostare l'indice del valore corrente di `threadIdx.x`, che ci dirà quale numero dell'embedding dobbiamo recuperare. Quindi, l'indice diventa `gpu_input_tokens[0] * 2048 + threadIdx.x`. Combiniamolo in un'implementazione del kernel:

```
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    input_embeddings[threadIdx.x] = embed_tokens[gpu_input_tokens[0] * 2048 + threadIdx.x];
}
```

Sembra a posto. Facciamolo funzionare per più di un singolo token in input. Cosa cambia? Volevamo eseguire più block — un block per token in input. Quindi dobbiamo tenerne conto nell'aritmetica degli indici. Come apparirà l'indice di output del thread corrente? Prima, era solo `threadIdx.x`. Ora dobbiamo tener conto che ci sono più token. Ogni token risulta in un embedding di dimensione 2048. Quindi, possiamo moltiplicare l'indice del block per 2048 e sommarlo a `threadIdx.x` per ottenere una posizione in un embedding per il token attualmente processato (e indice_token == indice_block). Quindi diventa:

```
int workIndex = threadIdx.x + blockIdx.x * 2048;
```

Non possiamo più hardcodare il primo token `gpu_input_tokens[0]`. Dato che indice_token == indice_block, possiamo sostituire `[0]` con `blockIdx.x`:

```
embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
```

Combinandolo insieme otteniamo:

```
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
}
```

Creiamo una funzione che è un wrapper sopra questa invocazione del kernel, così puoi eseguirla dal tuo codice C++. I kernel stanno in file `.cu`, per via della loro sintassi specifica, come il `<<<>>>` dove definisci la grid (block e thread).

```
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens)
{
    embeddingGatherKernel<<<???, 2048>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens);
}
```

Un attimo. Non conosciamo il numero di token. Non usiamo più strutture dati C++, come i vettori, quindi non possiamo leggere il numero di elementi di `gpu_input_tokens`. Dobbiamo passarlo esplicitamente a questa funzione:

```
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    embeddingGatherKernel<<<num_input_tokens, 2048>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens);
}
```

Ora sembra tutto a posto! Lanciamolo. Ti incoraggio a farlo davvero.

E poi?

...

[È una trappola!](https://www.youtube.com/watch?v=4F4qzPbcFiA) Il massimo di thread per block è 1024, almeno per la maggior parte delle GPU NVIDIA che potresti avere a casa. Nella funzione `checkGPUStatus()` stampiamo perfino questa informazione:

```
std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
```

Quindi, cosa possiamo fare ora? Possiamo lanciare 2 volte più block, oppure processare 2 numeri in ogni thread invece di uno solo. L'autore sceglie la seconda opzione — probabilmente sarà più veloce, non richiede il lancio di più thread e non richiede alcuna sincronizzazione tra thread né tanto meno tra la scrittura della memoria di output.

Abbiamo massimo 1024 thread per block. L'embedding ha 2048 numeri. Vogliamo processare due numeri dell'embedding per thread. Che opzioni abbiamo? Possiamo processare il numero del thread corrente e il numero accanto ad esso, oppure possiamo processare il numero del thread corrente e il numero nella stessa posizione nella seconda metà dell'embedding, quindi indice del thread corrente + 1024. Funzionerebbe il primo approccio? Il primo thread processerebbe il numero 0 e 1. Il secondo il 2 e il 3. Il terzo il 4 e il 5. Probabilmente funzionerebbe anche questo (?), ma con più aritmetica degli indici. La seconda opzione è di nuovo più facile. Basta aggiungere 1024 sia all'indice di input che a quello di output.

```
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
    input_embeddings[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
}
```

Eseguilo. Questa volta funzionerà.

Complimenti, hai finito il tuo primo kernel CUDA!

Torniamo dal calcolo e dalla programmazione a basso livello alla semantica/significato di quello che abbiamo appena fatto. Nota che, sebbene tutti questi embedding recuperati per i token in input abbiano un significato codificato al loro interno, non si "conoscono" tra loro. Non conoscono la propria posizione all'interno del testo fornito in input. Non sanno da quali token sono circondati. Non conoscono la storia della conversazione, ecc. Questa comprensione verrà costruita e memorizzata come proiezioni K e V.

## RMSNorm e riduzione parallela in CUDA

Guarda di nuovo la sequenza di operazioni nel nostro modello (sezione [Safetensors e il tuo modello](#safetensors-e-il-tuo-modello)). Dopo aver recuperato gli embedding per i nostri token, è il momento di [RMSNorm](https://arxiv.org/abs/1910.07467). A differenza del recupero degli embedding, è la prima operazione che gira nei layer. Il nostro modello, Llama 3.2 1B, ha 16 layer. RMSNorm prende gli embedding recuperati e — usando i pesi del modello per una rms norm `weights.input_layernorm[layer]` — esegue la funzione RMSNorm. RMSNorm è un'operazione che modifica tutti i numeri in un embedding. Per farlo, prima deve vedere tutti gli elementi e calcolare la loro somma di [radice quadratica media](https://en.wikipedia.org/wiki/Root_mean_square).

In base al paper, la formula è:

$$ \text{normalizzato}_i = \frac{a_i}{\text{RMS(a)}} \text{, dove RMS(a)}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a^2_i} $$

Scriviamo insieme questo kernel.

Per calcolare RMSNorm, dobbiamo prima calcolare `RMS(a)`. Richiede di passare per tutti i numeri, quindi ci servirà un po' di sincronizzazione tra i thread. Normalizziamo ogni embedding separatamente. Quindi di nuovo, lanciamo tanti block quanti sono i token in input. Gli embedding hanno 2048 numeri, ma 1024 è il massimo di thread per block, quindi useremo lo stesso trucco usato nel kernel di recupero degli embedding — processeremo due numeri all'interno di ogni thread. Ci serve un valore temporaneo, su cui possiamo scrivere i quadrati di ogni numero di un embedding. Potremmo usare una singola variabile per questo. Ogni thread all'interno di un block scriverebbe un quadrato del proprio numero su di essa. Il problema qui è la sincronizzazione delle letture e scritture sulla variabile condivisa — prevenire una race condition, assicurandoci di ottenere una somma corretta. C'è un altro approccio, dove creiamo un vettore temporaneo della stessa lunghezza del numero di thread in un block, e ogni thread scrive un quadrato del proprio valore nella posizione `threadIdx.x` di un vettore temporaneo. In questo modo, ci assicuriamo che nessun due thread proveranno a leggere o scrivere nella stessa posizione del vettore. Questa tecnica si chiama [riduzione parallela](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (tree reduction).

Creiamo un vettore. Usiamo la parola chiave `__shared__` per indicare che il vettore è condiviso tra tutti i thread di un block.

Una cosa sulla stabilità numerica prima di andare avanti — ricorda che i nostri dati sono in formato bfloat16. I principali punti di forza di bfloat16 sono un grande esponente — 8 bit, stessa dimensione di float (float32) — ma la mantissa è piccola (solo 7 bit contro i 10 bit di float16). Significa che per non perdere precisione nel nostro calcolo, possiamo usare un tipo più grande per calcolare effettivamente le cose dentro il kernel — cose come il quadrato di ogni numero. Useremo `float` per questo scopo. Trasformeremo (cast) ogni numero in float e dichiareremo anche il nostro buffer temporaneo come float.

```
__shared__ float rms_vector[1024];
rms_vector[threadIdx.x] = (float)input[threadIdx.x] * (float)input[threadIdx.x];
```

Volevamo calcolare due numeri per thread, quindi o rendiamo `rms_vector` più grande (2048 elementi) o aggiungiamo il quadrato di un numero distante 1024 indici al nostro `rms_vector[threadIdx.x]`. La seconda opzione è più facile da implementare e sarà probabilmente molto più veloce, perché avremo meno passaggi in una tree reduction — è più veloce ridurre a 1 elemento da 1024 elementi che da 2048 elementi.

```
__shared__ float rms_vector[1024];
rms_vector[threadIdx.x] = (float)input[threadIdx.x] * (float)input[threadIdx.x] + (float)input[threadIdx.x + 1024] * (float)input[threadIdx.x + 1024];
```

Funzionerebbe, se lanciassimo un solo block. Ma di nuovo, vogliamo essere in grado di processare più token, quindi lanceremo tanti block quanti sono i token. Per questo motivo, dobbiamo capire gli indici corretti degli elementi per il thread corrente. `threadIdx.x` è corretto come indice di `rms_vector`, perché `rms_vector` è sempre 1024 float. `threadIdx.x` non basterà come indice di input, perché abbiamo più token in input. Sappiamo che ogni token occupa 2048 `__nv_bfloat16`. Un indice di block ci dice l'indice di un token. Quindi, per spostarci al token corrente nell'input, dobbiamo moltiplicare `blockIdx.x` per la dimensione di un token — per 2048. Il nostro indice di input per il thread corrente è quindi:

```
int workIndex = threadIdx.x + blockIdx.x * 2048;
```

Ora, la parte di riduzione. L'idea è che ogni elemento i-esimo aggiunge a se stesso un elemento all'indice `self + i`. E poi, moltiplichiamo i per 2 e ripetiamo. Dopo ogni iterazione, dobbiamo assicurarci che tutti i thread finiscano di scrivere su `rms_vector` prima di passare all'iterazione successiva. CUDA ha [`__syncthreads()`](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#thread_synchronization), che possiamo usare per questo scopo, e lo mettiamo dopo aver scritto su `rms_vector`. Si può fare in un ciclo, ma inizieremo scrivendo tutto esplicitamente, così l'algoritmo di tree reduction risulterà più comprensibile. L'algoritmo finisce quando arriviamo a `self + 1024`. La somma di tutti gli elementi è memorizzata all'indice 0 — `rms_vector[0]`. Scriviamo il codice così puoi vederlo tu stesso. Per l'autore, molti pezzi non erano ovvi finché non li ha scritti lui stesso. Magari vuoi provarci anche tu, prima di leggere il codice?

Fai pure quello con cui ti senti più a tuo agio, e l'autore scriverà comunque la versione verbosa (ma corretta) del kernel RMSNorm:

```
__shared__ float rms_vector[1024];
int workIndex = threadIdx.x + blockIdx.x * 2048;
rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
__syncthreads();
```

Ogni thread in un block memorizza il quadrato del proprio numero in input sommato al quadrato del numero in input, spostato a destra di 1024 (coprendo la seconda parte del vettore di 2048, che non possiamo coprire lanciando 2048 thread in un block, per il limite CUDA di massimo 1024 thread per block). `rms_vector[0]` contiene 2 quadrati su 2048 numeri dell'embedding di input attualmente processato (per il token corrente) — indici `0` e `1024`.

Possiamo passare al primo passo della riduzione. Ogni secondo thread aggiungerà il numero successivo in `rms_vector`. Ricorda che ogni numero in `rms_vector` memorizza già due quadrati.

```
if (threadIdx.x % 2 == 0) {
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1];
}
__syncthreads();
```

`%` è l'operatore modulo ([divisione con resto](https://en.wikipedia.org/wiki/Euclidean_division)). Ora `rms_vector[0]` memorizza i quadrati dei numeri agli indici `0`, `1024` e `1`. Sappiamo che `1` conteneva già `1` e `1025`, quindi `rms_vector[0]` ha in realtà questi 4 quadrati: `0`, `1`, `1024` e `1025`.

Aumentiamo il "salto" di 2, così ora ogni 4° elemento aggiungerà un elemento distante 2 indici da sé:

```
if (threadIdx.x % 4 == 0) {
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 2];
}
__syncthreads();
```

Nota che l'indice che sommiamo a `threadIdx.x` è la metà di quello con cui facciamo il modulo (facciamo modulo per 4 e aggiungiamo l'elemento distante 2 posizioni). Pensa al perché lo facciamo così, e se la somma darebbe il risultato corretto se aggiungessimo lo stesso numero con cui facciamo il modulo.

Il processo continua finché non facciamo modulo di `threadIdx.x` per 1024:

```
if (threadIdx.x % 2 == 0)
{
    // ogni secondo elemento ha il proprio predecessore
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1];
}
__syncthreads();
if (threadIdx.x % 4 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 2];
}
__syncthreads();
if (threadIdx.x % 8 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 4];
}
__syncthreads();
if (threadIdx.x % 16 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 8];
}
__syncthreads();
if (threadIdx.x % 32 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 16];
}
__syncthreads();
if (threadIdx.x % 64 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 32];
}
__syncthreads();
if (threadIdx.x % 128 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 64];
}
__syncthreads();
if (threadIdx.x % 256 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 128];
}
__syncthreads();
if (threadIdx.x % 512 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 256];
}
__syncthreads();
if (threadIdx.x % 1024 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 512];
}
__syncthreads();
```

Ora `rms_vector[0]` contiene la somma di tutti i quadrati dei numeri dell'embedding di quel token. Per calcolare $\text{RMS(a)}$ dobbiamo dividerlo per la dimensione di un embedding (2048) e prenderne la radice quadrata: $\text{RMS(a)}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a^2_i}$.

```
if (threadIdx.x == 0)
{
    rms_vector[0] = sqrt(rms_vector[0] / 2048.0); // = RMS(a)
}
__syncthreads();
```

Ora tutti i thread possono leggere `rms_vector[0]` per recuperare $\text{RMS(a)}$. Finiamo il nostro kernel calcolando un valore normalizzato per i numeri del thread corrente (quello che corrisponde a threadIdx.x e a threadIdx.x + 1024). Ricorda che vogliamo usare `float` quando facciamo matematica che potrebbe essere affetta dalla precisione limitata di `bfloat16`. Quindi, trasformiamo (cast) tutti i numeri in `float` e, prima di scrivere nella memoria di output, li trasformiamo di nuovo in `__nv_bfloat16`.

```
output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);

output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
```

Quasi finito!

C'è un problema con queste due righe. Ricordiamo come viene usato $\text{RMS(a)}$ quando calcoliamo un valore normalizzato: $\text{normalizzato}_i = \frac{a_i}{\text{RMS(a)}}$. Se $\text{RMS(a)}$ è 0, allora questa divisione diventa una divisione per 0 e in C++ otterremo NaN o infinito, entrambi significano che abbiamo perso, e anche un solo NaN corromperà l'inferenza e il nostro motore produrrà spazzatura o andrà in crash. Per evitare che questo succeda, l'LLM di riferimento che usiamo ha un parametro epsilon, che definisce quale valore aggiungiamo al risultato di RMSNorm.

```
(input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
```

In questo modo, preveniamo la divisione per zero e la propagazione di NaN. Aggiorniamo semplicemente la riga dove calcoliamo $\text{RMS(a)}$:

```
if (threadIdx.x == 0)
{
    rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5); // = RMS(a)
}
__syncthreads();
```

Complimenti per aver finito il tuo kernel CUDA RMSNorm! Per essere meno verbosi, possiamo sostituire la scrittura manuale di ogni iterazione della riduzione con un ciclo `for`, e abbiamo finito qui:

```
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
        rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);
    }
    __syncthreads();

    output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
    output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
}
```

## RoPE

Nel nostro modello di riferimento, l'operazione successiva a RMSNorm è [RoPE](https://arxiv.org/pdf/2104.09864), un modo di codificare la posizione dei token nello stato nascosto (embedding). Una descrizione molto accessibile del positional encoding con RoPE è [qui, di Christopher Fleetwood](https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding).

Prova a scriverlo da solo, se hai energia per farlo. Se no, ecco il kernel finito dell'autore. Ci sono alcune cose che si potrebbero ottimizzare, come alcuni valori che possono essere precalcolati una volta e poi condivisi — vedi theta e angoli.

```
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
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

*< TODO dell'autore: descrivere in maggior dettaglio >*

## Connessioni residue

È una tecnica semplice usata in molti modelli di deep learning diversi, dove sommi gli input agli output, così i tuoi dati in input non vengono mai persi del tutto. Si fa per stabilità nel training. Vedi [maggiori informazioni qui](https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55/). Dal nostro punto di vista è solo la somma elemento per elemento di due vettori della stessa dimensione.

```
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
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

## cublasGemmEx

[Moltiplicazione di matrici](https://en.wikipedia.org/wiki/Matrix_multiplication) è una delle operazioni principali usate nel deep learning, in particolare negli LLM di grandi dimensioni. Una matrice è una tabella di numeri. Ha righe e colonne. Una singola riga e una singola colonna si chiamano vettore — una sequenza di numeri. La moltiplicazione di matrici usa due matrici, A e B, in input, e produce la matrice C in output. La matrice A ha dimensioni (M, K). La matrice B ha dimensioni (K, N). Sia la matrice A che B condividono la stessa dimensione K. In altre parole, le righe della matrice A hanno la stessa lunghezza delle colonne della matrice B. Quando moltiplichi A per B, ottieni una nuova matrice C con dimensioni (M, N):

$$A (M,K) \times B(K,N)=C(M,N)$$

Ogni elemento della matrice C ($c_{ij}$) è un [prodotto scalare](https://en.wikipedia.org/wiki/Dot_product) della riga i-esima di A e della colonna j-esima di B. Il prodotto scalare è una somma di tutte le coppie, dove ogni coppia è il risultato della moltiplicazione dei numeri dalla riga i-esima di A con i numeri dalla riga j-esima di B agli stessi indici all'interno dei loro vettori (sia la riga che la colonna sono vettori):

$$c_{ij} = a_{i0}b_{0j} + a_{i1}b_{1j}+...+a_{ik}b_{kj}=\sum_{x=0}^{k} a_{ix}b_{xj}$$

Torniamo ai modelli linguistici di grandi dimensioni. La moltiplicazione di matrici avviene quando si calcolano l'attention e le proiezioni Q, K e V. L'hardware più popolare per calcolarla in modo efficiente sono le GPU NVIDIA. Forniscono una libreria importante, [cuBLAS](https://developer.nvidia.com/cublas), che ti permette di eseguire calcoli di algebra lineare ad alte prestazioni sulle loro GPU, inclusa la moltiplicazione di matrici, tramite la funzione [cublasGemmEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex).

## Il trucco di trasposizione column-major → row-major

**TL;DR: se i tuoi dati sono in formato row-major e userai cuBLAS, allora imposta il flag di trasposizione a `CUBLAS_OP_T` per le matrici che non sono ancora trasposte, e `CUBLAS_OP_N` per le matrici che sono trasposte nella tua formula.**

Ora la derivazione e la comprensione:

Il problema con `cublasGemmEx` è che si aspetta che tu fornisca le matrici in [formato column-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order). E gli LLM, come Llama 3.2 1B Instruct, sono distribuiti in formato row-major.

[![diagramma column e row major](https://github.com/jmaczan/tiny-vllm/raw/main/assets/column-row-major.png)](/jmaczan/tiny-vllm/blob/main/assets/column-row-major.png)

Si scopre che non dobbiamo modificare il formato dei dati per usare le funzioni di moltiplicazione di matrici di cuBLAS. Tutto grazie a queste proprietà:

$$[A^T]_{ij}=[A]_{ji} \qquad C^T=B^T \times A^T \qquad (A^T)^T=A$$

Il "$^T$" significa che trasponiamo la matrice. Trasporre una matrice trasforma le colonne in righe, e le righe in colonne. Quando memorizzi la matrice in formato row-major, e cuBLAS la legge in formato column-major, è l'equivalente di trasporre la matrice.

Vediamo un esempio per capirlo meglio: vogliamo calcolare $C = A \times B$, dove A ha dimensioni (5, 2048) e B ha dimensioni (512, 2048). La dimensione desiderata di C è (5, 512). In questo momento, le dimensioni di A e B sono incompatibili: $A(5, 2048)$ e $B(512, 2048)$. Ricordi che per ottenere $C(M,N)$ ci servono $A(M,K)$ e $B(K,N)$? In altre parole, la seconda dimensione di A e la prima dimensione di B devono essere uguali. Per ottenerlo, dobbiamo trasporre B. La formula diventa ora: $C = A \times B^T$. Le dimensioni ora vanno bene: $A(5,2048) \times B(2048, 512) = C(5, 512)$. Okay, quindi ora vorremmo usare cuBLAS per calcolare C.

Ma cuBLAS si aspetta il formato column-major di A e B. Row-major trasposto ci darà column-major. Quindi trasponiamo la formula $C = A \times B^T$, usando la proprietà $C^T=B^T \times A^T$ e otteniamo $C^T = (B^T)^T \times A^T$. Dalla terza proprietà delle matrici sopra, sappiamo che una trasposizione di una trasposizione è uguale alla matrice originale, quindi semplifichiamo $(B^T)^T$ a semplicemente $B$. La formula finale è: $C^T = B \times A^T$. Verifichiamo che le dimensioni siano ancora corrette. $B (512, 2048) \times A^T(2048, 5) = C^T(512, 5)$. Le dimensioni del risultato sembrano l'inverso di quello che volevamo ottenere — (5, 512) — ma nota che stiamo ancora parlando di $C^T$. La $C$ effettiva, l'output di `cublasGemmEx`, non è trasposta, quindi la dimensione finale è corretta (5, 512). Ora il codice:

```
cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, KV_DIM, num_active_slots, EMBEDDING_LENGTH, &k_proj_alpha, weights.w_k[layer], CUDA_R_16BF, EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH, &k_proj_beta, k_proj_batched_buffer, CUDA_R_16BF, KV_DIM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

L'autore vuole anticipare l'ultima confusione che potresti avere se scavi davvero nel codice. I flag `CUBLAS_OP_T` e `CUBLAS_OP_N` dicono a cuBLAS quali matrici trasporre. E abbiamo appena derivato la formula $C^T=B \times A^T$, quindi perché ora diciamo a cuBLAS di trasporre la prima matrice $B$? Per capirlo, pensa di nuovo a column-/row-major. Dalla prospettiva di cuBLAS, la nostra $B$ row-major è trasposta $B^T$, perché cuBLAS la legge come se fosse column-major. Quindi dobbiamo dire a cuBLAS di trasporla, per riottenere la $B$ che abbiamo derivato. Analogamente, dato che abbiamo derivato che il secondo argomento dovrebbe essere $A^T$, e cuBLAS legge la $A$ row-major come una $A^T$ column-major, allora non trasponiamola di nuovo, perché è così che volevamo fornirla a `cublasGemmEx`. C.V.D. :D

> L'autore pubblicherà questa sezione in una forma leggermente diversa in [Paged Out! Issue #9, nell'articolo "The cuBLAS transposition trick"](https://pagedout.institute/).

## Prefill vs decode

Un fatto interessante sull'inferenza LLM è che non è esattamente lo stesso processo per il primo token predetto rispetto a tutti i successivi. Per predire il primo token, devi processare tutti i token in input. I token in input sono il prompt dell'utente o la cronologia della chat. Tutto il calcolo necessario per ottenere il primo token predetto si chiama *prefill*. Tutto ciò che succede dopo si chiama *decode*.

La maggior parte del calcolo, come proiezione Q, attention, score di attention, feed-forward (MLP), viene scartato non appena passato all'operazione successiva — sia in prefill che in decode. Il modello mentale utile è che l'unica cosa che preservi in ogni fase dell'inferenza LLM è la proiezione K, la proiezione V e qual è l'ultimo token generato. Tutto qui. Ci sono implicazioni interessanti — potresti fermare l'inferenza, copiare le tue proiezioni K e V e l'ultimo token generato, riavviare il server, caricarle nel server e usare l'ultimo token generato come input, e otterresti la stessa identica predizione del token successivo che avresti ottenuto nell'istanza originale del server. L'autore spera che qualcuno metta alla prova questa sua affermazione e la testi davvero — fatelo sapere se lo fate :D

## Perché esiste il KV cache

Possiamo riusare alcune parti dei risultati del calcolo per predire i token successivi. Non sei obbligato a riusarli, ma non cambiano, quindi ricalcolarli continuamente è puro spreco. Sai già che l'unico dato che va avanti nel calcolo è la proiezione K, V e l'ultimo token generato. Se generiamo 1 token alla volta, sia K che V sono vettori di bfloat16 e l'ultimo token generato è un singolo intero. Se generiamo più token contemporaneamente — in altre parole, se facciamo batching — sia K che V sono matrici di bfloat16 e gli ultimi token generati sono un vettore di interi.

Quando processiamo un token, sia in prefill che in decode, dal punto di vista dei dati che preserviamo (proiezioni K, V e ultimo token generato) appare così:

0. ...
1. Calcola proiezione K usando l'ultimo token generato
2. Memorizzala
3. Calcola proiezione V usando l'ultimo token generato
4. Memorizzala
5. ...
6. Usa tutte le proiezioni K e tutte le proiezioni V per calcolare l'attention
7. ...
8. Genera nuovo token
9. Memorizzalo come ultimo token generato

Diciamo che non memorizziamo la proiezione K e V per il token corrente. Significherebbe che dovremmo calcolare tutte le proiezioni K e V per il token corrente e per tutti i precedenti prima di poter calcolare l'attention per il token corrente. Di nuovo, puro spreco. Ecco perché memorizziamo le proiezioni K e V. È solo una registrazione di tutte le proiezioni K e V precedenti. Non la modifichi durante l'inferenza LLM. Ci aggiungi solamente, ad ogni token processato. Il nome di questo storage di proiezioni K e V è KV cache.

## Attention

L'attention è una parte importante dell'inferenza LLM. È dove fai molta moltiplicazione di matrici usando le proiezioni Q, K e V calcolate prima. Una formula base per lo scaled dot-product attention, che viene dal paper [Attention is all you need](https://arxiv.org/pdf/1706.03762), è:

$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

*< TODO dell'autore: questa sezione è difficile da scrivere, ma l'autore vuole farla bene e utile per te. Mette il codice qui per ora, e vuole tornare a scriverla per bene più avanti >*

```
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
    // i * input_tokens.size() * input_tokens.size(), perché attn scores è (32, num_tok, num_tok)
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

A differenza della multi-head attention, uno dei primi metodi di attention, nella [group-query attention (GQA)](https://arxiv.org/pdf/2305.13245) più head di query condividono la stessa head di key e value. Nel nostro caso è così: 4 head di query usano la stessa 1 head di key e value. Vedi il codice con il calcolo delle head K/V sopra.

*< TODO dell'autore: descrivere in maggior dettaglio >*

## SiLU

[SiLU](https://arxiv.org/pdf/1702.03118) è una funzione di attivazione usata nel nostro LLM di riferimento. Introduce "non-linearità" in un modello. Significa che i pesi possono essere azzerati quando non servono. Aiuta nel training dei modelli. Quasi tutti i modelli di machine learning ce l'hanno nella loro architettura, anche i più semplici multi-layer perceptron. Anzi, per quanto l'autore ricordi, i modelli non riuscivano a generalizzare abbastanza bene senza funzioni di attivazione. SiLU è simile a ReLU, ma quando i valori negativi si avvicinano a 0, non vengono azzerati, ma ottengono invece un piccolo valore negativo.

```
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

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) è una funzione che normalizza tutti gli elementi in un vettore. Questa è la prima versione, "sequenziale", di un softmax. Deriveremo e implementeremo la versione "online" più avanti.

$$ \sigma(v)=\frac{e^{v_i}}{\sum_{j=1}{K}e^{v_j}} $$

Il modo in cui puoi implementarlo è molto simile a RMSNorm che abbiamo implementato prima.

```
__global__ void softmaxKernel(__nv_bfloat16 *input, int num_tokens)
{
    // softmax per head
    // potrebbe sprecare molta memoria hardcodando la dimensione qui, ma non si può usare direttamente num_tokens
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
        max_val = row[0]; // così non serve allocare un altro valore shared per max_val
    }
    __syncthreads();

    // trasforma in exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // ora possiamo calcolare la somma numericamente stabile, stesso pattern - tree reduction
    // riusando la memoria di row
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
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

## Maschera causale

Ogni token può fare attention solo verso i token precedenti. Vedi [questa buona e diretta spiegazione](https://outcomeschool.com/blog/causal-masking-in-attention), così puoi codificarlo da solo. È davvero utile guardare i diagrammi.

*< TODO dell'autore: scrivere di più >*

```
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
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

## Argmax

Scegli il token con il punteggio più alto.

```
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

```
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              HIDDEN_DIM,       // m
              1,                // n
              EMBEDDING_LENGTH, // k
              &gate_alpha,
              weights.mlp_gate_proj[layer],
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              rms_norms,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              &gate_beta,
              gate,
              CUDA_R_16BF,
              HIDDEN_DIM,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);

// (1, 2048) * (2048, 8192) -> (1, 8192)
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              HIDDEN_DIM,       // m
              1,                // n
              EMBEDDING_LENGTH, // k
              &up_alpha,
              weights.mlp_up_proj[layer],
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              rms_norms,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              &up_beta,
              up,
              CUDA_R_16BF,
              HIDDEN_DIM,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);

silu(gate, up, 1);

down = buf_2048_2;
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              EMBEDDING_LENGTH, // m
              1,                // n
              HIDDEN_DIM,       // k
              &down_alpha,
              weights.mlp_down_proj[layer],
              CUDA_R_16BF,
              HIDDEN_DIM,
              gate,
              CUDA_R_16BF,
              HIDDEN_DIM,
              &down_beta,
              down,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);
```

## Riuso dei buffer

A volte, un buffer che allochi ha la stessa dimensione di un buffer che verrà usato più avanti nel codice. E a volte, quando il secondo buffer inizia a servire dopo che i dati del primo buffer non servono più (è già stato usato in un calcolo e non verrà più usato altrove), possiamo usare il primo buffer per scrivere i dati che scriveremmo nel secondo buffer. In questo modo, possiamo allocare meno memoria. Significa che riusiamo lo stesso buffer in due punti diversi. Possiamo farlo in sicurezza una volta confermato, tramite un'analisi del ciclo di vita, che i cicli di vita di questi due buffer non si sovrappongono. Vedi come funziona per `buf_2048_1` e `buf_2048_2` in [`src/main.cpp`](https://github.com/jmaczan/tiny-vllm/blob/main/src/main.cpp).

*< TODO dell'autore: scrivere di più e spiegare l'analisi del ciclo di vita >*

## Batching statico

Invece di processare una richiesta alla volta, processiamo N richieste. I pro: throughput maggiore (più richieste utente processate contemporaneamente). I contro: latenza più alta (tutti i prompt nel batch devono aspettare che il prompt più lungo finisca di essere processato, prima di essere restituiti all'utente). In pratica devi, in ogni punto dove assumevamo un singolo token processato alla volta, processarne multipli. Lo stesso vale per il prefill — per più prompt, ecc.

*< TODO dell'autore: scrivere di più >*

## Batching continuo

Risolve il problema di dover aspettare il prompt più lungo nel batch. Lo risolve avendo slot in un batch. Riempi i prompt negli slot. Una volta finita la generazione in un dato slot, il risultato viene restituito all'utente. Un prompt in attesa in coda viene selezionato per lo slot appena liberato. Il prompt passa per il prefill (tutti gli altri elementi nel batch aspettano finché non è finito). Una volta finito il prefill, il batching continua con tutti gli elementi del batch, incluso quello nuovo.

*< TODO dell'autore: scrivere di più >*

## Online softmax

*< TODO dell'autore: scrivere di più e scrivere la derivazione matematica >*

[Online Softmax: Cohere Labs talk - Slides](https://docs.google.com/presentation/d/1f4Ys4es1l8bfUFN1bJuy12c3zHIPelcNGgVgbyb1zac/edit)

[From Boltzmann-Gibbs distribution to numerically stable online softmax on GPU with proofs](https://jedrzej.maczan.pl/boltzmann_gibbs_to_softmax_on_gpu.pdf)

## Paged Attention

*In arrivo!*

Idea di gestione della memoria dai sistemi operativi, usata nell'inferenza LLM.

> ⚠️ **AVVISO — quanto segue in questa sezione non fa parte del README originale.** Le due righe qui sopra ("In arrivo!" e la frase sulla gestione della memoria) sono **tutto** ciò che l'autore aveva scritto al momento in cui ho letto il repo. Quanto segue è la **mia interpretazione**, basata su come funziona PagedAttention nel paper originale di vLLM (Kwon et al., 2023) e su quanto abbiamo già discusso insieme nel nostro corso teorico. **Non è testo di Maczan.** Quando l'autore pubblicherà la sua versione ufficiale di questa sezione, confrontala con quanto scritto qui.

Nel KV cache "ingenuo" che abbiamo implementato finora, ogni richiesta riserva un blocco di memoria **contiguo**, dimensionato sulla lunghezza massima di sequenza possibile (es. `MAX_PROMPT_LEN` + token generabili). Questo funziona bene per un motore che gestisce poche richieste alla volta, ma diventa un problema serio quando servi molte richieste simultanee con lunghezze molto diverse tra loro — lo scenario tipico di un motore da produzione come vLLM.

Il problema è duplice:

1. **Spreco per allocazione eccessiva**: se prenoti sempre la lunghezza massima per ogni richiesta, ma la maggior parte delle richieste reali è molto più corta, sprechi enormi quantità di memoria GPU "prenotata ma inutilizzata".
2. **Frammentazione**: richieste che iniziano e finiscono in momenti diversi, con lunghezze finali imprevedibili, lasciano "buchi" di memoria difficili da riutilizzare in modo efficiente con allocazioni contigue di dimensione variabile.

**L'idea di PagedAttention**, dal paper di Kwon et al. (2023, UC Berkeley), prende in prestito il concetto di **paginazione della memoria virtuale** dai sistemi operativi: invece di un unico blocco contiguo per richiesta, il KV cache viene diviso in tante **pagine** (o "blocchi") di dimensione fissa e piccola (es. spazio sufficiente per 16 token). Ogni richiesta ottiene le pagine di cui ha effettivamente bisogno, man mano che la sequenza cresce — non tutte in anticipo — e queste pagine **non devono essere fisicamente contigue in memoria**.

Una **tabella dei blocchi** (block table) per ogni richiesta tiene traccia di quali pagine fisiche, in che ordine, compongono la sequenza logica di quella richiesta — esattamente come una tabella delle pagine di un sistema operativo mappa indirizzi virtuali a indirizzi fisici.

## Paged KV cache

*In arrivo!*

> ⚠️ **AVVISO — anche qui, l'originale aveva solo "In arrivo!" senza altro testo.** Quanto segue è la mia interpretazione, non testo di Maczan.

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

*In arrivo!*

> ⚠️ **AVVISO — anche qui, l'originale aveva solo "In arrivo!" senza altro testo.** Quanto segue è la mia interpretazione, non testo di Maczan.

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

Il costo aggiuntivo rispetto al kernel "semplice" è la lettura indiretta tramite `block_table` (un ulteriore accesso a memoria per risolvere "quale pagina fisica"), che introduce un pattern di accesso meno prevedibile — un compromesso simile a quello discusso nel nostro corso a proposito del gather indicizzato per il routing MoE: si scambia un po' di efficienza di accesso puro per una gestione della memoria molto più flessibile ed efficiente in aggregato. In pratica, l'implementazione reale di vLLM ottimizza pesantemente questo accesso (con letture allineate per warp, prefetching delle pagine, ecc.) proprio per minimizzare questo overhead.

---

Jędrzej Maczan, 2026, Apache License 2.0

*Fine della traduzione. Repo originale: [github.com/jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm). Traduzione italiana e sezioni integrative su PagedAttention a cura di Claude — vedi gli avvisi nel testo per distinguere le due fonti.*
