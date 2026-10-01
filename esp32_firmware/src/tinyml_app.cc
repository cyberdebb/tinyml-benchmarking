#include <cstdint>
#include <cstdio>

#include "esp_cpu.h"
#include "esp_private/esp_clk.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_log.h"

#include "tinyml_common.h"
#include "tinyml_model.h"

namespace {

constexpr char TAG[] = "tinyml";

void init_console()
{
    setvbuf(stdin, nullptr, _IONBF, 0);
    uart_driver_install(UART_NUM_0, 2 * kLineSize, 0, 0, nullptr, 0);
    uart_vfs_dev_use_driver(UART_NUM_0);
}

// Timing uses the CPU cycle counter (CCOUNT), the same method as the
// STM32 firmware (DWT->CYCCNT), instead of esp_timer, which has 1 us
// resolution and its own call overhead. CCOUNT is per core: the main task
// is pinned to CPU0 (CONFIG_ESP_MAIN_TASK_AFFINITY_CPU0), so start and end
// are read on the same core.
uint32_t esp32_timer_get_cycles()
{
    return esp_cpu_get_cycle_count();
}

// CCOUNT wraps every 2^32 cycles (~17.9 s at 240 MHz), so intervals are
// taken as an unsigned cycle difference, which stays correct across the
// wrap, and only then converted to microseconds.
long long esp32_cycles_to_us(uint32_t start, uint32_t end)
{
    const uint32_t cycles = end - start;
    return static_cast<long long>(
        (static_cast<uint64_t>(cycles) * 1000000u) /
        static_cast<uint64_t>(esp_clk_cpu_freq()));
}

}

extern "C" void tinyml_app_main(void)
{
    init_console();

    ESP_LOGI(TAG, "Model: %s", kModelName);
    if (!init_model()) {
        ESP_LOGE(TAG, "Model initialization failed");
        return;
    }

    print_model_info();
    ESP_LOGI(TAG, "Ready. Send one 256-sample CSV beat + 4 RR values per line.");

    // static so the main task stack is not blown
    static char line[kLineSize];
    static float beat[kInputSamples];
    static float rr[kRrFeatures];

    while (std::fgets(line, sizeof(line), stdin) != nullptr) {
        if (is_blank(line)) {
            continue;
        }
        if (!parse_beat(line, beat, rr)) {
            ESP_LOGW(TAG, "Expected 256 samples + 4 RR values, comma-separated");
            continue;
        }

        // Measured on the device: preprocessing + inference only, with the
        // serial transfer left out.
        const uint32_t filter_start = esp32_timer_get_cycles();
        filter_sos(beat);
        const uint32_t inference_start = esp32_timer_get_cycles();
        const int prediction = classify(beat, rr);
        const uint32_t end = esp32_timer_get_cycles();

        const long long filter_us = esp32_cycles_to_us(filter_start, inference_start);
        const long long inference_us = esp32_cycles_to_us(inference_start, end);

        // Machine-readable result line for the host script.
        std::printf("RESULT,%d,%lld,%lld\n", prediction, inference_us, filter_us);
        std::fflush(stdout);
    }

    ESP_LOGE(TAG, "stdin closed, leaving tinyml_app_main");
}
