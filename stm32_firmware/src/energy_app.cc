// Current/energy test firmware (<model>_energy environments), used instead
// of tinyml_app.cc. See power_logger/README.md for the wiring and the test.
//
// During the test the MCU is powered only through the INA226 of the power
// logger (JP5 IDD jumper removed, IN- on the MCU side) and the ST-LINK USB
// is unplugged, so there is no serial link: the firmware runs on its own,
// forever:
//
//   idle kIdleMs  ->  burst of back-to-back inferences for kBurstMs  -> ...
//
// Idle is the MCU at rest: WFI (Sleep mode, CPU clock stopped) woken only by
// the 1 ms SysTick, clocks and peripherals as configured by board_init()
// (216 MHz). That is the baseline the logger subtracts.
//
// The sync pin (kSyncPin -> power logger GPIO4) toggles right before each
// inference. The logger counts both edges in hardware (PCNT), so it knows
// how many inferences ran (N) and when the burst started and ended, without
// any serial link. Toggling (instead of a pulse per inference) needs no
// minimum pulse width, so nothing is added to the inference loop but one
// register write.
//
// The beats are the DS2 beats in models/energy_beats.h (ml/src/energy_beats.py),
// filtered once at boot: what is measured is classify(), the same part the
// DS2 test reports as inference_us.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "board.h"
#include "tinyml_common.h"
#include "tinyml_model.h"

#include "models/energy_beats.h"

#define DWT_LAR_UNLOCK_KEY 0xC5ACCE55

namespace {

// PF13: D7 on the ST Zio connector CN10.
GPIO_TypeDef *const kSyncPort = GPIOF;
constexpr uint16_t kSyncPin = GPIO_PIN_13;
// Idle must be longer than the logger's end-of-burst gap (AUTO, 1000 ms by
// default) plus enough time for the baseline.
constexpr uint32_t kIdleMs = 5000;
constexpr uint32_t kBurstMs = 3000;

// Filtered copies of the beats, in RAM like the beats of the DS2 test.
float beats[kEnergyBeatCount][kInputSamples];

// Keeps the compiler from dropping classify() calls whose result is unused
// (the RF and SVM are pure functions).
volatile int prediction_sink = 0;

void stm_timer_init()
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = DWT_LAR_UNLOCK_KEY;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void init_sync_pin()
{
    __HAL_RCC_GPIOF_CLK_ENABLE();
    HAL_GPIO_WritePin(kSyncPort, kSyncPin, GPIO_PIN_RESET);
    GPIO_InitTypeDef gpio = {};
    gpio.Pin = kSyncPin;
    gpio.Mode = GPIO_MODE_OUTPUT_PP;
    gpio.Pull = GPIO_NOPULL;
    // Slow edges: less ringing on the jumper wire to the logger.
    gpio.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(kSyncPort, &gpio);
}

// The ST-LINK is unpowered during the test: USART3 TX idles high and would
// feed current into its RX pin. Back to analog (high impedance) instead.
void release_uart()
{
    HAL_UART_DeInit(&huart3);
    HAL_GPIO_DeInit(GPIOD, GPIO_PIN_8 | GPIO_PIN_9);
}

void idle_ms(uint32_t ms)
{
    const uint32_t start = HAL_GetTick();
    while (HAL_GetTick() - start < ms) {
        __WFI();
    }
}

// Back-to-back inferences for kBurstMs; `level` is the sync pin state,
// kept across bursts.
void run_burst(uint32_t &level)
{
    // CYCCNT wraps every ~19.9 s at 216 MHz; the burst is much shorter, so
    // the unsigned difference is enough.
    const uint32_t duration_cycles = kBurstMs * (SystemCoreClock / 1000u);
    const uint32_t start = DWT->CYCCNT;
    int index = 0;
    int sink = 0;
    do {
        level ^= 1u;
        kSyncPort->BSRR = level ? kSyncPin : static_cast<uint32_t>(kSyncPin) << 16;
        sink += classify(beats[index], kEnergyRr[index]);
        index = index + 1 < kEnergyBeatCount ? index + 1 : 0;
    } while (DWT->CYCCNT - start < duration_cycles);
    prediction_sink = sink;
}

}

extern "C" void tinyml_app_main(void)
{
    stm_timer_init();
    init_sync_pin();

    // Only seen if the ST-LINK is connected (flashing / checking the board).
    std::printf("[energy] Model: %s (energy test firmware)\n", kModelName);
    if (!init_model()) {
        std::printf("[energy] Model initialization failed\n");
        return;
    }
    print_model_info();

    for (int i = 0; i < kEnergyBeatCount; ++i) {
        std::memcpy(beats[i], kEnergyBeats[i], sizeof(beats[i]));
        filter_sos(beats[i]);
    }
    std::printf("[energy] %d beats. Cycle: idle %lu ms, burst %lu ms, sync on PF13 (D7)\n",
                kEnergyBeatCount, static_cast<unsigned long>(kIdleMs),
                static_cast<unsigned long>(kBurstMs));
    std::fflush(stdout);
    release_uart();

    uint32_t level = 0;
    for (;;) {
        idle_ms(kIdleMs);
        run_burst(level);
    }
}
