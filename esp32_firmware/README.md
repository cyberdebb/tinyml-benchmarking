# Firmware do ESP32-S3

Firmware que roda os seis classificadores de ECG no **ESP32-S3 DevKitC-1** e
o script que mede, pela serial, as métricas de classificação e o tempo de
inferência no conjunto de teste (DS2). Também tem o firmware do teste de
corrente e energia, usado com o [power logger](../power_logger/README.md).

Visão geral do projeto: [README principal](../README.md).

## Sumário

- [Hardware e software](#hardware-e-software)
- [Estrutura](#estrutura)
- [Environments](#environments)
- [Como o firmware funciona](#como-o-firmware-funciona)
- [Teste do DS2 (classificação e tempo)](#teste-do-ds2-classificação-e-tempo)
- [Teste de corrente e energia](#teste-de-corrente-e-energia)
- [Tamanho e memória dos modelos](#tamanho-e-memória-dos-modelos)
- [Problemas comuns](#problemas-comuns)

## Hardware e software

| Item | Valor |
|---|---|
| Placa | ESP32-S3 DevKitC-1 (`board = esp32-s3-devkitc-1`) |
| CPU | Xtensa LX7 dual-core a **240 MHz**, FPU de precisão simples |
| Flash | 8 MB, modo QIO a 80 MHz; partição de app de 7 MB (`partitions.csv`) |
| Caches | instrução 32 KB, dados 64 KB (linha de 64 B) |
| Framework | ESP-IDF **5.4.1**, pelo PlatformIO (`platform = espressif32`) |
| Inferência | `espressif/esp-tflite-micro` 1.4.0 com kernels **ESP-NN** 1.3.2 (`CONFIG_NN_OPTIMIZED`) |
| Compilador | **GCC 14.2.0** (Xtensa), `-O2` (`CONFIG_COMPILER_OPTIMIZATION_PERF`), asserts desligados |
| Serial | UART0 a 115200 baud |

A configuração está em `sdkconfig.defaults`. Os arquivos
`sdkconfig.<environment>` são gerados a partir dela na primeira compilação.
Depois de mudar um `*.defaults`, apague os `sdkconfig.<environment>` para que
sejam gerados de novo.

## Estrutura

```
esp32_firmware/
├── platformio.ini              um environment por modelo (+ os _energy)
├── sdkconfig.defaults          CPU, flash, caches, otimização, ESP-NN
├── sdkconfig.energy.defaults   o que muda no firmware de energia
├── partitions.csv
├── include/
│   ├── tinyml_common.h         formato da entrada (256 amostras + 4 RR, 5 classes)
│   ├── tinyml_model.h          interface comum: init_model(), classify(), print_model_info()
│   ├── beat_features.h         12 features morfológicas (MLP, RF, SVM)
│   ├── models/                 gerado pelo ml/pipeline.py (passo 2), fora do git
│   └── data/energy_beats.h     gerado pelo ml/pipeline.py (passo 2b), fora do git
├── src/
│   ├── main.c                  app_main() -> tinyml_app_main()
│   ├── tinyml_app.cc           teste do DS2: lê batimentos da serial e responde RESULT
│   ├── energy_app.cc           teste de energia: repouso / rajada, pino de sincronismo
│   ├── tinyml_common.cc        filtro SOS e leitura da linha CSV
│   ├── beat_features.cc        extração das features morfológicas
│   ├── model_cnn.cc            as três CNNs (TFLite Micro)
│   ├── model_mlp.cc            MLP (TFLite Micro)
│   ├── model_rf.cc             Random Forest (if/else do emlearn)
│   └── model_svm.cc            SVM (arrays C)
└── scripts/
    ├── benchmark_serial.py     teste do DS2 pela serial
    └── model_size.py           tamanho real do Random Forest no binário
```

## Environments

Cada modelo é um *environment* do PlatformIO. O `src/CMakeLists.txt` escolhe o
`model_<modelo>.cc` e o app pela variável `TINYML_MODEL`.

| Environment | Modelo | App |
|---|---|---|
| `cnn_rn` | CNN ResNet-1D | teste do DS2 (`tinyml_app.cc`) |
| `cnn_mn2` | CNN MobileNetV2-1D | teste do DS2 |
| `cnn_mn3` | CNN MobileNetV3-1D | teste do DS2 |
| `mlp` | MLP | teste do DS2 |
| `rf` | Random Forest | teste do DS2 |
| `svm` | SVM | teste do DS2 |
| `<modelo>_energy` | o mesmo modelo | teste de energia (`energy_app.cc`) |

```bash
cd esp32_firmware
pio run -e cnn_rn -t upload          # compila e grava
pio run -e cnn_rn_energy -t upload   # firmware de energia do mesmo modelo
pio device monitor                   # 115200 baud
```

Antes de compilar, os modelos precisam estar em `include/models/`: rode os
passos 1 e 2 do `ml/pipeline.py` (veja o [README principal](../README.md)).

## Como o firmware funciona

1. **Inicialização:** carrega o modelo e, nas redes, reserva uma *arena* de
   128 KB na RAM interna para o TFLite Micro.
2. **Linha `INFO`:** a placa envia `INFO,<modelo>,<bytes do modelo>,<bytes de RAM>`
   e depois `Ready.`.
3. **Para cada linha recebida pela serial** (256 amostras + 4 valores de RR,
   separados por vírgula):
   - `filter_sos()` aplica o passa-banda de 0,5–45 Hz na janela;
   - `classify()` roda o modelo (no MLP, RF e SVM, inclui extrair as features);
   - a placa responde `RESULT,<classe>,<inference_us>,<filter_us>`.

**Medição de tempo:** pelo contador de ciclos da CPU (`CCOUNT`), não pelo
`esp_timer`, que tem resolução de 1 µs e um custo próprio por chamada. A
tarefa principal fica presa ao CPU0 (`CONFIG_ESP_MAIN_TASK_AFFINITY_CPU0`),
então o início e o fim são lidos no mesmo núcleo. É o mesmo método da STM32
(`DWT->CYCCNT`).

## Teste do DS2 (classificação e tempo)

Com o environment normal gravado (por exemplo `cnn_rn`) e o USB conectado:

```bash
python esp32_firmware/scripts/benchmark_serial.py COM3 cnn_rn               # DS2 inteiro
python esp32_firmware/scripts/benchmark_serial.py COM3 cnn_rn --resume      # continua uma execução parada
python esp32_firmware/scripts/benchmark_serial.py COM3 cnn_rn --beats 500   # amostra rápida
```

- **Reset automático:** o script reinicia a placa pelas linhas DTR/RTS, espera
  o `Ready.` e envia os batimentos do DS2, os mesmos pacientes da avaliação do
  treino.
- **Duração:** são horas, porque cada batimento tem ~2 KB a 115200 baud.
- **Interrupções:** cada resultado é salvo no CSV assim que chega. Se a
  execução parar (erro de serial, Ctrl+C, reset), rode o mesmo comando com
  `--resume`.
- **Só refazer o relatório:** se todos os batimentos já estiverem no CSV, o
  `--resume` não envia nada. A porta precisa estar no comando, mas não é
  usada se existir o `<modelo>_info.json`.
- **`--beats N`:** usa uma amostra estratificada de N batimentos e salva com
  o sufixo `_sample<N>`, então nunca sobrescreve uma execução completa.

### Resultados

Em `esp32_firmware/results/inference/`:

| Arquivo | Conteúdo |
|---|---|
| `<modelo>_serial.csv` | uma linha por batimento: índice, registro, classe real, classe prevista, `inference_us`, `filter_us` |
| `<modelo>_report.txt` | relatório AAMI por classe e por registro, acurácia, sensibilidade, especificidade, F1 e F-beta (macro sobre N, S, V, F e sobre N, S, V), F1 por classe, tempos e vazão |
| `<modelo>_info.json` | tamanho e RAM do modelo reportados pela placa |

Formato do resumo no fim do relatório:

```
==================================================
  Inference test: CNN ResNet-1D on the ESP32-S3
  Model size: <KB>
  Model RAM:  <KB>
  Beats: <batimentos>
  Accuracy:      <...>
  Sensitivity:   <...> (Macro Avg N, S, V, F)
  ...
  Inference time (us): <média> +- <desvio> [<mín> .. <máx>]
  SOS filter (us):     <média> +- <desvio> [<mín> .. <máx>]
  Throughput:    <...> inferences/second
==================================================
```

Os tempos aparecem como média ± desvio padrão [mín .. máx] sobre os
batimentos, em µs, a resolução da medida.

## Teste de corrente e energia

Os environments `<modelo>_energy` usam `energy_app.cc` em vez do teste do DS2.
Durante a medição a placa é alimentada **só pelo INA226**, sem o cabo USB, e
roda sozinha, para sempre:

```
repouso 5 s  ->  rajada de inferências seguidas por 3 s  ->  repouso 5 s  -> ...
```

- **Repouso:** a tarefa principal fica bloqueada em `vTaskDelay()` e os dois
  núcleos ficam na tarefa *idle* do FreeRTOS (`waiti`), com a CPU a 240 MHz.
- **Sincronismo:** o **GPIO4** é invertido antes de cada inferência. O power
  logger conta as bordas e sabe quantas inferências rodaram e onde começa e
  termina cada rajada.
- **Batimentos:** os 32 batimentos do DS2 de `include/data/energy_beats.h`
  (8 de cada classe N, S, V, F) são filtrados uma vez na inicialização. Só o
  `classify()` é medido, a mesma parte que o teste do DS2 reporta como
  `inference_us`.
- **Watchdog:** o `sdkconfig.energy.defaults` desliga a checagem de *idle* do
  watchdog no CPU0. Como a rajada nunca libera a CPU, o aviso dele cairia
  dentro da janela de medição.

### Conexões da placa ESP32-S3

| Sinal/componente | Origem (pino) | Destino (pino) | Observações |
|---|---|---|---|
| Alimentação da placa (saída do *shunt*) | INA226 IN- | ESP32-S3 3V3 | Único caminho de energia do chip; cabo USB do ESP32-S3 desconectado durante a medição |
| Terra comum (GND) | ESP32-S3 GND | GND comum | Nó único com o ESP32 auxiliar e o INA226 |
| Sincronismo | ESP32-S3 GPIO4 (saída) | ESP32 auxiliar GPIO4 (entrada) | Invertido antes de cada inferência; o logger conta as bordas para marcar início e fim de cada rajada |

As conexões comuns (INA226 e ESP32 auxiliar) e o passo a passo estão em
[`power_logger/README.md`](../power_logger/README.md).

> **Atenção:** para gravar, **tire o fio IN- e o fio de sincronismo** e
> conecte o USB. O regulador da placa e o 3V3 do logger não podem ficar
> ligados juntos.

## Tamanho e memória dos modelos

O firmware reporta na linha `INFO` o tamanho do modelo e a RAM que ele usa:

| Modelo | Tamanho (flash) | RAM |
|---|---|---|
| CNNs e MLP | `sizeof()` do array gerado a partir do `.tflite` | *arena* do TFLite Micro efetivamente usada (`arena_used_bytes()`) |
| SVM | `SVM_MODEL_BYTES`: soma dos arrays C (vetores de suporte e coeficientes duais em int16, interceptos e índices) | buffer de valores do kernel (um `float` por vetor de suporte) |
| Random Forest | medido no binário pelo `scripts/model_size.py` | nenhuma além da pilha |

**Random Forest:** o emlearn gera a floresta como código `if/else`, então não
há um array cujo `sizeof()` possa ser usado. A placa só tem uma estimativa
(nós × 8 bytes). O `benchmark_serial.py` troca essa estimativa pelo tamanho
real, lido de `.pio/build/rf`. Para isso, o `firmware.elf` diz quais funções
da floresta sobraram depois do *linker* e o objeto de `model_rf.cc` dá o
tamanho delas. No Xtensa, isso inclui as seções `.literal`, onde ficam os
limiares em `float`. Basta ter compilado o environment `rf` (`pio run -e rf`)
e ter o `pyelftools` instalado.

## Problemas comuns

- **`Timed out waiting for the firmware to become ready`:** porta errada, ou
  o monitor serial ainda está aberto. Feche o monitor ou aperte o RESET.
- **`Dataset cache not found`:** rode o treino (passo 1 do `ml/pipeline.py`)
  antes, porque o script lê o DS2 do cache.
- **`models/... .h: No such file`:** os modelos ainda não foram copiados
  para `include/models/` (passo 2 do pipeline).
- **Mudou um `sdkconfig*.defaults` e nada mudou:** apague os
  `sdkconfig.<environment>` gerados e compile de novo.
