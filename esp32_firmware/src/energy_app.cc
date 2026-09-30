// Current/energy test firmware (<model>_energy environments), used instead
// of tinyml_app.cc. See power_logger/README.md for the wiring and the test.
//
// During the test the board is powered only through the INA226 of the power
// logger and has no serial link, so it runs on its own, forever:
//
//   idle kIdleMs  ->  burst of back-to-back inferences for kBurstMs  -> ...
//
// Idle is the MCU at rest: the main task blocked in vTaskDelay(), both
// cores in the FreeRTOS idle task (waiti, clock-gated until the next tick),
// CPU at 240 MHz. That is the baseline the logger subtracts.
//
// The sync pin (kSyncPin -> power logger GPIO4) toggles right before each
// inference. The logger counts both edges in hardware (PCNT), so it knows
// how many inferences ran (N) and when the burst started and ended, without
// any serial link. Toggling (instead of a pulse per inference) needs no
// minimum pulse width, so nothing is added to the inference loop but one
// register write.
//
// The beats are the DS2 beats in data/energy_beats.h (ml/src/energy_beats.py),
// filtered once at boot: what is measured is classify(), the same part the
// DS2 test reports as inference_us.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "driver/gpio.h"
#include "esp_cpu.h"
#include "esp_log.h"
#include "esp_private/esp_clk.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tinyml_common.h"
#include "tinyml_model.h"

#include "data/energy_beats.h"

namespace {

constexpr char TAG[] = "energy";

constexpr gpio_num_t kSyncPin = GPIO_NUM_4;
// Idle must be longer than the logger's end-of-burst gap (AUTO, 1000 ms by
// default) plus enough time for the baseline. The burst never yields; the
// watchdog's idle check on CPU0 is off in this build
// (sdkconfig.energy.defaults), so it can be longer than 5 s.
constexpr uint32_t kIdleMs = 5000;
constexpr uint32_t kBurstMs = 3000;

// Filtered copies of the beats, in RAM like the beats of the DS2 test.
float beats[kEnergyBeatCount][kInputSamples];

// Keeps the compiler from dropping classify() calls whose result is unused
// (the RF and SVM are pure functions).
volatile int prediction_sink = 0;

void init_sync_pin()
{
    gpio_config_t config = {};
    config.pin_bit_mask = 1ull << kSyncPin;
    config.mode = GPIO_MODE_OUTPUT;
    config.pull_up_en = GPIO_PULLUP_DISABLE;
    config.pull_down_en = GPIO_PULLDOWN_DISABLE;
    config.intr_type = GPIO_INTR_DISABLE;
    gpio_config(&config);
    gpio_set_level(kSyncPin, 0);
}

uint32_t cycles_per_ms()
{
    return static_cast<uint32_t>(esp_clk_cpu_freq() / 1000);
}

// Back-to-back inferences for kBurstMs; `level` is the sync pin state,
// kept across bursts.
void run_burst(uint32_t &level)
{
    // CCOUNT wraps every ~17.9 s at 240 MHz; the burst is much shorter, so
    // the unsigned difference is enough.
    const uint32_t duration_cycles = kBurstMs * cycles_per_ms();
    const uint32_t start = esp_cpu_get_cycle_count();
    int index = 0;
    int sink = 0;
    do {
        level ^= 1u;
        gpio_set_level(kSyncPin, level);
        sink += classify(beats[index], kEnergyRr[index]);
        index = index + 1 < kEnergyBeatCount ? index + 1 : 0;
    } while (esp_cpu_get_cycle_count() - start < duration_cycles);
    prediction_sink = sink;
}

}

extern "C" void tinyml_app_main(void)
{
    init_sync_pin();

    ESP_LOGI(TAG, "Model: %s (energy test firmware)", kModelName);
    if (!init_model()) {
        ESP_LOGE(TAG, "Model initialization failed");
        return;
    }
    print_model_info();

    for (int i = 0; i < kEnergyBeatCount; ++i) {
        std::memcpy(beats[i], kEnergyBeats[i], sizeof(beats[i]));
        filter_sos(beats[i]);
    }
    ESP_LOGI(TAG, "%d beats. Cycle: idle %u ms, burst %u ms, sync on GPIO%d",
             kEnergyBeatCount, static_cast<unsigned>(kIdleMs),
             static_cast<unsigned>(kBurstMs), static_cast<int>(kSyncPin));

    // Nothing is printed from here on: a UART write right after a burst
    // would fall inside the logger's measurement window.
    uint32_t level = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(kIdleMs));
        run_burst(level);
    }
}
