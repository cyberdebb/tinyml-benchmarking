#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "board.h"
#include "tinyml_common.h"
#include "tinyml_model.h"

#define DWT_LAR_UNLOCK_KEY 0xC5ACCE55

namespace {

void stm_timer_init()
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = DWT_LAR_UNLOCK_KEY;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t stm_timer_get_cycles()
{
    return DWT->CYCCNT;
}

// CYCCNT wraps every 2^32 cycles (~19.9 s at 216 MHz), so intervals are
// taken as an unsigned cycle difference, which stays correct across the
// wrap, and only then converted to microseconds.
unsigned long stm_cycles_to_us(uint32_t start, uint32_t end)
{
    const uint32_t cycles = end - start;
    return static_cast<unsigned long>(
        (static_cast<uint64_t>(cycles) * 1000000u) / SystemCoreClock);
}

char *uart_read_line(UART_HandleTypeDef *huart, char *buffer, size_t buffer_size)
{
    size_t index = 0;
    while (index < buffer_size - 1) {
        uint8_t byte;
        if (HAL_UART_Receive(huart, &byte, 1, HAL_MAX_DELAY) != HAL_OK) {
            return nullptr;
        }
        buffer[index++] = (char)byte;
        if (byte == '\n') {
            break;
        }
    }
    buffer[index] = '\0';
    return buffer;
}

}

extern "C" void tinyml_app_main(void)
{
    stm_timer_init();

    std::printf("[tinyml] Model: %s\n", kModelName);
    if (!init_model()) {
        std::printf("[tinyml] Model initialization failed\n");
        return;
    }

    print_model_info();
    std::printf("[tinyml] Ready. Send one 256-sample CSV beat + 4 RR values per line.\n");

    // static so the stack is not blown
    static char line[kLineSize];
    static float beat[kInputSamples];
    static float rr[kRrFeatures];

    while (uart_read_line(&huart3, line, sizeof(line)) != nullptr) {
        if (is_blank(line)) {
            continue;
        }
        if (!parse_beat(line, beat, rr)) {
            std::printf("[tinyml] Expected 256 samples + 4 RR values, comma-separated\n");
            continue;
        }

        // Measured on the device: preprocessing + inference only, with the
        // serial transfer left out.
        const uint32_t filter_start = stm_timer_get_cycles();
        filter_sos(beat);
        const uint32_t inference_start = stm_timer_get_cycles();
        const int prediction = classify(beat, rr);
        const uint32_t end = stm_timer_get_cycles();

        const unsigned long filter_us = stm_cycles_to_us(filter_start, inference_start);
        const unsigned long inference_us = stm_cycles_to_us(inference_start, end);

        // Machine-readable result line for the host script. %lu, not %lld:
        // newlib-nano's printf has no long long support.
        std::printf("RESULT,%d,%lu,%lu\n", prediction, inference_us, filter_us);
        std::fflush(stdout);
    }

    std::printf("[tinyml] stdin closed, leaving tinyml_app_main\n");
}
