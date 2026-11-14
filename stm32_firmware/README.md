# Firmware da STM32 NUCLEO-F767ZI

Firmware que roda os seis classificadores de ECG na **STM32 NUCLEO-F767ZI**
(Arm Cortex-M7) e o script que mede, pela serial, as métricas de
classificação e o tempo de inferência no conjunto de teste (DS2). Também tem
o firmware do teste de corrente e energia, usado com o
[power logger](../power_logger/README.md).

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
| Placa | NUCLEO-F767ZI (`board = nucleo_f767zi`) |
| MCU | STM32F767ZI, Arm Cortex-M7 a **216 MHz**, FPU de precisão dupla (FPv5-D16) |
| Memória | 2 MB de flash, 512 KB de RAM; caches de instrução e de dados ligadas |
| Clock | HSI interno de 16 MHz → PLL → 216 MHz (com *over-drive*) |
| Framework | STM32Cube HAL, pelo PlatformIO (`platform = ststm32`) |
| Inferência | TensorFlow Lite Micro em `lib/tflm`, com kernels **CMSIS-NN** (`-D CMSIS_NN`) |
| Compilador | **GCC 12.3.1** (`toolchain-gccarmnoneeabi ~1.120301.0`), `-O2`, `-std=gnu++17` |
| Serial | USART3 (PD8/PD9), ligada à porta COM virtual do ST-LINK, a 115200 baud |

### Por que GCC 12.3.1

A versão do GCC inicialmente disponível para a plataforma era a 8.2.1. Adotou-se
a versão 12.3.1 para aproximar as versões das *toolchains* das duas placas (o
ESP32-S3 usa o GCC 14.2.0); não havia um pacote do GCC 14 para o STM32 no
servidor do PlatformIO. A versão é fixada em `platformio.ini`:

```ini
platform_packages =
    toolchain-gccarmnoneeabi @ ~1.120301.0
```

### FPU por hardware (`fpu.py`)

O builder `stm32cube` do PlatformIO só passa `-mcpu=cortex-m7`. Assim, o GCC
usa ponto flutuante por software e cada operação em `float` (filtro SOS,
features, `expf` da SVM, comparações do RF) vira uma chamada de biblioteca.
O script `fpu.py` (`extra_scripts = pre:fpu.py`) adiciona
`-mfloat-abi=hard -mfpu=fpv5-d16` ao compilador **e ao linker**. Os flags não
podem ir só em `build_flags`, que chegam apenas ao compilador: sem eles, o
link falha com `uses VFP register arguments`.

### Clock pelo HSI

O clock vem do HSI interno, e não do MCO de 8 MHz do ST-LINK (HSE em
*bypass*). O `stm32f7xx_hal_conf.h` padrão assume `HSE_VALUE = 25 MHz`, o que
deixaria o `SystemCoreClock` errado e, com ele, a conversão de ciclos para µs.

## Estrutura

```
stm32_firmware/
├── platformio.ini              um environment por modelo (+ os _energy), GCC 12.3.1
├── fpu.py                      FPU por hardware no compilador e no linker
├── include/
│   ├── board.h                 inicialização da placa (clock, caches, USART3)
│   ├── tinyml_common.h         formato da entrada (256 amostras + 4 RR, 5 classes)
│   ├── tinyml_model.h          interface comum: init_model(), classify(), print_model_info()
│   ├── beat_features.h         12 features morfológicas (MLP, RF, SVM)
│   ├── models/                 gerado pelo ml/pipeline.py (passo 2), fora do git
│   └── data/energy_beats.h     gerado pelo ml/pipeline.py (passo 2b), fora do git
├── lib/tflm/                   TensorFlow Lite Micro + CMSIS-NN (código-fonte)
├── src/
│   ├── main.c                  board_init() -> tinyml_app_main()
│   ├── board.c                 clock de 216 MHz, caches, USART3, printf pela serial
│   ├── tinyml_app.cc           teste do DS2: lê batimentos da serial e responde RESULT
│   ├── energy_app.cc           teste de energia: repouso (WFI) / rajada, pino de sincronismo
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

O código dos modelos, das features e do filtro é o mesmo do ESP32-S3. Mudam
só a inicialização da placa, a serial e a medição de tempo.

## Environments

O `build_src_filter` de cada environment escolhe o `model_<modelo>.cc` e o app.

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
cd stm32_firmware
pio run -e cnn_rn -t upload          # compila e grava pelo ST-LINK
pio run -e cnn_rn_energy -t upload   # firmware de energia do mesmo modelo
pio device monitor                   # 115200 baud
```

Antes de compilar, os modelos precisam estar em `include/models/`: rode os
passos 1 e 2 do `ml/pipeline.py` (veja o [README principal](../README.md)).

## Como o firmware funciona

1. **Inicialização da placa:** `board_init()` liga as caches, configura o
   clock de 216 MHz e a USART3, e faz o `stdout` ir para a serial linha a
   linha.
2. **Modelo:** nas redes, reserva uma *arena* de 128 KB para o TFLite Micro.
3. **Linha `INFO`:** a placa envia `INFO,<modelo>,<bytes do modelo>,<bytes de RAM>`
   e depois `Ready.`.
4. **Para cada linha recebida** (256 amostras + 4 valores de RR):
   - `filter_sos()` aplica o passa-banda de 0,5–45 Hz na janela;
   - `classify()` roda o modelo (no MLP, RF e SVM, inclui extrair as features);
   - a placa responde `RESULT,<classe>,<inference_us>,<filter_us>`.

**Medição de tempo:** pelo contador de ciclos do Cortex-M7 (`DWT->CYCCNT`),
convertido para µs com o `SystemCoreClock`. É o mesmo método do ESP32-S3
(`CCOUNT`). O contador dá a volta a cada ~19,9 s a 216 MHz, então os
intervalos são calculados como diferença sem sinal, o que continua correto
na virada.

## Teste do DS2 (classificação e tempo)

Com o environment normal gravado (por exemplo `cnn_rn`) e o USB do ST-LINK
conectado:

```bash
python stm32_firmware/scripts/benchmark_serial.py COM5 cnn_rn               # DS2 inteiro
python stm32_firmware/scripts/benchmark_serial.py COM5 cnn_rn --resume      # continua uma execução parada
python stm32_firmware/scripts/benchmark_serial.py COM5 cnn_rn --beats 500   # amostra rápida
```

> **Aperte o botão RESET (preto) da Nucleo** quando o script mostrar
> `Waiting for the firmware. Press the RESET button...`. Ao contrário do
> ESP32-S3, a porta COM do ST-LINK não reinicia a placa pelas linhas DTR/RTS.

- **Duração:** são horas, porque cada batimento tem ~2 KB a 115200 baud.
- **Interrupções:** cada resultado é salvo no CSV assim que chega. Se a
  execução parar, rode o mesmo comando com `--resume`.
- **Só refazer o relatório:** se todos os batimentos já estiverem no CSV, o
  `--resume` não envia nada.
- **`--beats N`:** usa uma amostra estratificada de N batimentos, salva com o
  sufixo `_sample<N>`.

### Resultados

Em `stm32_firmware/results/inference/`:

| Arquivo | Conteúdo |
|---|---|
| `<modelo>_serial.csv` | uma linha por batimento: índice, registro, classe real, classe prevista, `inference_us`, `filter_us` |
| `<modelo>_report.txt` | relatório AAMI por classe e por registro, acurácia, sensibilidade, especificidade, F1 e F-beta (macro sobre N, S, V, F e sobre N, S, V), F1 por classe, tempos e vazão |
| `<modelo>_info.json` | tamanho e RAM do modelo reportados pela placa |

O resumo começa com `Inference test: <modelo> on the STM32 NUCLEO-F767ZI` e
mostra os tempos como média ± desvio padrão [mín .. máx] sobre os batimentos,
em µs.

## Teste de corrente e energia

Os environments `<modelo>_energy` usam `energy_app.cc` em vez do teste do DS2.
Durante a medição o **chip** STM32 é alimentado **só pelo INA226**, pelo
jumper de medição de corrente JP5 (IDD), e o firmware roda sozinho, para
sempre:

```
repouso 5 s  ->  rajada de inferências seguidas por 3 s  ->  repouso 5 s  -> ...
```

- **Repouso:** laço de `WFI` (modo *Sleep*: o clock da CPU para e o SysTick
  de 1 ms acorda o chip), com os clocks e periféricos a 216 MHz.
- **Sincronismo:** o pino **PF13** (D7 no conector CN10) é invertido antes de
  cada inferência. O power logger conta as bordas.
- **Serial desligada:** a USART3 é desativada (pinos em alta impedância) para
  não injetar corrente no ST-LINK.
- **Batimentos:** os 32 batimentos do DS2 de `include/data/energy_beats.h`
  são filtrados uma vez na inicialização. Só o `classify()` é medido.

### Conexões da placa STM32 NUCLEO-F767ZI

| Sinal/componente | Origem (pino) | Destino (pino) | Observações |
|---|---|---|---|
| Alimentação do STM32 (saída do *shunt*) | INA226 IN- | NUCLEO JP5 (IDD), pad do lado do STM32 | Jumper IDD removido; pad do lado do regulador sem conexão; ST-LINK permanece conectado por USB, fora do domínio medido pelo INA226 |
| Terra comum (GND) | NUCLEO GND | GND comum | Nó único com o ESP32 auxiliar e o INA226 (use um GND, não o AGND) |
| Sincronismo | NUCLEO PF13 (D7) (saída) | ESP32 auxiliar GPIO4 (entrada) | Invertido antes de cada inferência; o logger conta as bordas para marcar início e fim de cada rajada |

**Por que o USB do ST-LINK fica conectado:**

- **Sem conflito de fontes:** com o JP5 retirado, o regulador da Nucleo
  alimenta só o ST-LINK e o resto da placa. O chip STM32 recebe energia só
  pelo INA226 (UM1974, seção 7.7).
- **Sem o USB, a placa fica reiniciando:** o ST-LINK sem alimentação
  interfere no reset (NRST) do chip (UM1974, seção 7.4.2).
- **Qual pad do JP5 é qual:** com o jumper fora e só o USB conectado, o pad do
  lado do regulador mede ~3,3 V e o do lado do STM32 mede ~0–0,8 V. Com o IN-
  no pad errado, o `--check` do logger mostra **corrente negativa**.
- **Chip de Ethernet:** ele envia um clock de 50 MHz para um pino da STM32,
  o que soma uma corrente constante ao repouso. A subtração do repouso cancela
  essa corrente no Einf, mas o repouso absoluto fica um pouco acima do consumo
  do chip sozinho.

> **Para gravar:** coloque o jumper JP5 de volta, desconecte o IN- e o fio de
> sincronismo e grave pelo USB do ST-LINK.

As conexões comuns (INA226 e ESP32 auxiliar) e o passo a passo estão em
[`power_logger/README.md`](../power_logger/README.md).

## Tamanho e memória dos modelos

O firmware reporta na linha `INFO` o tamanho do modelo e a RAM que ele usa:

| Modelo | Tamanho (flash) | RAM |
|---|---|---|
| CNNs e MLP | `sizeof()` do array gerado a partir do `.tflite` | *arena* do TFLite Micro efetivamente usada (`arena_used_bytes()`) |
| SVM | `SVM_MODEL_BYTES`: soma dos arrays C (vetores de suporte e coeficientes duais em int16, interceptos e índices) | buffer de valores do kernel (um `float` por vetor de suporte) |
| Random Forest | medido no binário pelo `scripts/model_size.py` | nenhuma além da pilha |

**Random Forest:** a floresta é código `if/else`, então a placa só tem uma
estimativa (nós × 8 bytes). O `benchmark_serial.py` troca essa estimativa pelo
tamanho real das funções da floresta no `firmware.elf` de `.pio/build/rf`. No
ARM, o tamanho do símbolo já inclui o *literal pool* com os limiares. Basta
ter compilado o environment `rf` e ter o `pyelftools` instalado.

## Problemas comuns

- **O script não sai de `Waiting for the firmware`:** aperte o RESET da
  Nucleo. Confira também se a porta é a do ST-LINK (*STMicroelectronics
  STLink Virtual COM Port*).
- **`init mode failed` ao gravar:**
  - o jumper JP5 precisa estar colocado nos dois pinos;
  - o IN- precisa estar solto;
  - se o firmware de energia já estiver gravado, segure o RESET, rode o
    upload e solte ao aparecer `Uploading`.
- **`uses VFP register arguments` no link:** o `fpu.py` não foi aplicado.
  Confira o `extra_scripts = pre:fpu.py` no `platformio.ini`.
- **Tempos muito maiores que o esperado:** verifique se a compilação usou o
  GCC 12.3.1 e a FPU por hardware. Sem a FPU, todo `float` vira chamada de
  biblioteca.
