# Teste de corrente e energia (power logger)

Modo extra, ao lado do teste do DS2 (`scripts/benchmark_serial.py`). Para
cada MCU (ESP32-S3 e STM32 NUCLEO-F767ZI) e para cada modelo (CNN, MLP, RF,
SVM), separadamente, mede:

- **Einf**: energia por inferência, com *baseline subtraction*:
  `Einf = (E_burst − P_idle × T) / N`. É a energia da inferência acima do
  custo do MCU em repouso (idle).
- **Iinst**: corrente durante a inferência, como média em blocos. Por burst,
  `I_inf` é a corrente média enquanto o MCU infere e `dI_inf` é o quanto ela
  fica acima do idle. Com `--profile`, também sai o perfil no tempo, com um
  ponto a cada ~0,34 ms (uma conversão do INA226, ou `--block` conversões).
  Na CNN (~55 ms por inferência) isso dá o formato de cada inferência. No
  MLP e no RF, que são mais rápidos que uma conversão, só a média do bloco
  tem sentido.

Os resultados ficam em `<mcu>_firmware/results/`, junto com os do DS2.

## Como funciona

Durante a medição a placa testada (DUT) é alimentada **só pelo INA226**, sem
USB (os diagramas de ligação). Então ela não recebe batimentos pela serial:
roda o firmware de energia (`<modelo>_energy`), que tem 32 batimentos do DS2
gravados nele e repete para sempre:

```
idle 5 s  ->  burst: inferências seguidas por 3 s  ->  idle 5 s  -> ...
```

- **Idle** é o MCU parado com os clocks ligados. No ESP32-S3, é a tarefa
  bloqueada em `vTaskDelay` (os núcleos ficam em `waiti`). Na STM32, é um
  laço de `WFI` (modo Sleep, acordado pelo SysTick).
- **Pino de sincronismo:** antes de cada inferência, a DUT inverte o pino. O
  ESP32 do logger conta as bordas em hardware (PCNT), então sabe exatamente
  quantas inferências (N) rodaram e onde começa e termina cada burst.
- **Medição de cada burst:** o logger integra a potência no burst e subtrai o
  idle medido logo antes dele. Isso cancela deriva, o offset do INA226 e o
  consumo fixo da placa (LEDs, reguladores).
- **Tempo de inferência:** `t_inf = (última borda − primeira borda) / (N − 1)`.

Por isso é preciso **um fio a mais** além dos diagramas: o de sincronismo,
da DUT para o GPIO4 do logger. É a mesma técnica do EnergyRunner do MLPerf
Tiny, que usa um GPIO de timestamp.

## Ligações

Comum aos dois setups (logger = "ESP32 extra"):

| Fio | Origem | Destino |
|---|---|---|
| Alimentação do logger | Notebook, USB | ESP32 extra |
| VCC do INA226 | ESP32 extra 3V3 | INA226 VCC |
| GND do INA226 | ESP32 extra GND | INA226 GND (mesmo nó do GND da DUT) |
| SDA / SCL | ESP32 extra GPIO21 / GPIO22 | INA226 SDA / SCL |
| ALE (alert) | — | não conectado |
| IN+ | ESP32 extra 3V3 | INA226 IN+ |
| VBS | — | ponte direta com IN- |
| **SYNC (novo)** | pino da DUT (abaixo) | **ESP32 extra GPIO4** |

ESP32-S3 (DUT):

| Fio | Origem | Destino |
|---|---|---|
| IN- | INA226 IN- | pino 3V3 do ESP32-S3 |
| GND | ESP32-S3 GND | GND comum |
| SYNC | ESP32-S3 **GPIO4** | ESP32 extra GPIO4 |
| USB do ESP32-S3 | — | **desconectado durante a medição** |

STM32 NUCLEO-F767ZI (DUT):

| Fio | Origem | Destino |
|---|---|---|
| IN- | INA226 IN- | pad "STM32" do JP5 (lado do chip, jumper removido) |
| GND | Nucleo GND | GND comum |
| SYNC | **PF13** (D7 no conector CN10) | ESP32 extra GPIO4 |
| USB do ST-LINK | — | **desconectado durante a medição** (só para gravar) |
| Pad "fonte" do JP5 | — | sem uso |

O GND precisa ser comum entre logger, INA226 e DUT (o SYNC é referenciado a
ele).

## Passo a passo

### 1. Uma vez: gerar os arquivos e gravar o logger

1. Treine os modelos e copie os arquivos para os firmwares, como já é feito
   para o DS2 (passos 1 e 2 do `ml/pipeline.py`). O pipeline agora também gera
   os batimentos do teste, `include/data/energy_beats.h`, nos dois firmwares
   (passo 2b, `write_energy_test_beats`). Se os modelos já estiverem
   treinados e copiados, basta rodar:

   ```
   python ml/src/energy_beats.py
   ```

2. Grave o logger (ESP32 extra ligado no notebook, INA226 ligado):

   ```
   cd power_logger
   pio run -t upload
   pio device monitor        # 921600 baud; deve aparecer "# INA226 at 0x40" e "READY"
   ```

   No monitor, digite `READ` + Enter para ver corrente, tensão e bordas de
   sync. Feche o monitor antes de rodar os scripts.

### 2. Para cada MCU e cada modelo

Exemplo com o ESP32-S3 e a CNN. Troque `cnn` por `mlp`, `rf` ou `svm`, e
`esp32` por `stm32`.

1. **Gravar o firmware de energia na DUT** (a DUT precisa do USB para isso):
   - ESP32-S3: **tire o fio IN- do 3V3 do ESP32-S3** (o regulador da placa e
     o 3V3 do logger não podem ficar ligados juntos), conecte o USB do
     ESP32-S3 e rode:
     ```
     cd esp32_firmware
     pio run -e cnn_energy -t upload
     ```
   - STM32: com o **jumper JP5 colocado** e o IN- desconectado, conecte o USB
     do ST-LINK e rode:
     ```
     cd stm32_firmware
     pio run -e cnn_energy -t upload
     ```
   - Ou pelo pipeline: `build_and_upload_firmware('cnn', esp32_firmware_directory, energy=True)`.

2. **Montar para a medição:** desconecte o USB da DUT. Na STM32, tire o
   jumper JP5. Religue o IN- (3V3 do ESP32-S3 ou pad "STM32" do JP5) e
   confira o fio SYNC e o GND. Ligue o logger no notebook: a DUT liga junto,
   alimentada pelo INA226.

3. **Conferir as ligações** (opcional, recomendado na primeira vez):
   ```
   python power_logger/scripts/benchmark_energy.py COM6 esp32 cnn --check
   ```
   `COM6` é a porta do **logger**. Devem aparecer:
   - a corrente da placa (dezenas de mA);
   - uma tensão de ~3,3 V;
   - `sync_edges` saltando a cada ~8 s, que são os bursts.

4. **Rodar o teste:**
   ```
   python power_logger/scripts/benchmark_energy.py COM6 esp32 cnn --profile
   ```
   - São 10 bursts medidos (`--bursts`) mais 1 descartado (`--warmup`),
     cerca de 1,5 min.
   - `--profile` salva o perfil de corrente. Vale para a CNN e a SVM; no MLP
     e no RF pode deixar sem, ou usar `--block 10` para um perfil mais leve.

5. Repita para os outros modelos e para a outra placa.

O teste do DS2 continua igual: grave o ambiente normal (`pio run -e cnn -t
upload`) e rode `scripts/benchmark_serial.py`.

### 3. Resultados

Em `esp32_firmware/results/` ou `stm32_firmware/results/`:

- **`<modelo>_energy.csv`**: uma linha por burst. As colunas principais são:
  - `n_inf` e `t_inf_us`: número de inferências e tempo de cada uma;
  - `E_inf_uJ`: **Einf**; `E_inf_sigma_uJ` é o piso de ruído dela;
  - `I_idle_mA` e `P_idle_mW`: o baseline (idle logo antes do burst);
  - `I_inf_mA` e `P_inf_mW`: **corrente e potência durante a inferência**,
    como média em blocos;
  - `dI_inf_mA` e `dP_inf_mW`: o mesmo, acima do idle;
  - `E_total_uJ`, `E_idle_uJ` e `E_net_uJ`: energia na janela, a parte do
    idle e a diferença entre as duas.
- **`<modelo>_energy_report.txt`**: média ± desvio sobre os bursts, com o
  erro padrão da média de Einf. Também compara o `t_inf` com o tempo de
  inferência do teste do DS2, se ele já tiver rodado.
- **`<modelo>_current_profile.csv`** (com `--profile`): colunas `burst`,
  `t_us`, `I_mA`, `dI_mA` e `P_mW`, a corrente de cada burst no tempo
  (Iinst por média em blocos).

## Problemas comuns

- **`Timed out waiting for the logger`:** a porta está errada (é a do ESP32
  extra) ou o INA226 não responde. O logger mostra o scan I2C e tenta de
  novo a cada 1 s. Se só chegar lixo, o script lê o logger também a 115200
  e diz o motivo (linhas `[DIAG]`).
- **`I2C bus held low` / `I2C bus timeout` / `I2C hardware timeout`:** SDA ou
  SCL está preso em 0, e o ESP32 não consegue nem começar uma transferência.
  Quase sempre é o INA226 sem alimentação: os pull-ups do módulo vão para o
  VCC dele e puxam as linhas para 0. Também pode ser SDA/SCL em pinos
  trocados ou em curto com o GND. Confira:
  - VCC do INA226 no 3V3 do ESP32 extra e GND no GND;
  - SDA no GPIO21 e SCL no GPIO22;
  - com o ESP32 extra ligado, SDA e SCL medem ~3,3 V em relação ao GND.
- **`No burst from the board`:**
  - a DUT não está com o firmware `_energy`;
  - o fio SYNC ou o GND não está ligado;
  - a DUT não está alimentada.

  O `--check` mostra qual desses é: sem corrente, a DUT está sem
  alimentação; com corrente, mas `sync_edges` parado, é o SYNC ou o
  firmware.
- **STM32 sem bursts, com o ST-LINK desconectado:** confira com o `--check`
  se a corrente muda a cada ~8 s. Se ficar sempre igual, o MCU não está
  rodando. Pode ser o reset segurado pelo ST-LINK sem alimentação: teste o
  firmware antes com o ST-LINK ligado e o jumper JP5 colocado.
- **`profile points dropped`:** o perfil com `--block 1` passou da serial.
  Use `--block 2` ou mais.
- **Comandos manuais:** o logger também aceita comandos no monitor serial,
  documentados no cabeçalho de `src/power_logger.cc`:
  - `AUTO <bursts>`;
  - `BASELINE`, `START` e `STOP` (modo manual);
  - `STREAM` e `BLOCK` (perfil);
  - `CFG` (tempos de conversão do INA226);
  - `RSHUNT` (resistor shunt; o padrão é 0,1 Ω, o "R100" dos módulos).
