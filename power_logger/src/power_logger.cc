// Power logger for the TinyML benchmark.
//
// Runs on a second ESP32 ("ESP32 extra") with an INA226 in series with the
// supply of the board under test (DUT): the 3V3 pin of the ESP32-S3, or the
// MCU side of the IDD jumper JP5 of the NUCLEO-F767ZI (power_logger/README.md):
//
//   ESP32 extra 3V3 -> INA226 IN+ -> shunt -> IN- (= VBS) -> DUT supply
//   ESP32 extra GPIO21/GPIO22 -> INA226 SDA/SCL, ALERT not connected
//   DUT sync pin -> ESP32 extra GPIO4 (AUTO mode)
//
// VBS is tied to IN-, so the bus voltage is the voltage at the DUT and
// P = V_bus * I is the power drawn by the DUT alone.
//
// Metrics
// -------
// Baseline subtraction: the DUT is measured at rest (idle) and while it runs
// N inferences back to back. Integrating over a window of length T that
// contains the inferences:
//
//   E_total = integral of P dt
//   E_net   = E_total - P_idle * T      (energy above the idle floor)
//   E_inf   = E_net / N                 (energy per inference, Einf)
//
// Idle time inside the window adds ~0 to E_net, so the window only has to
// enclose the inferences, not be aligned with them. The same subtraction
// cancels the INA226 offset error and the static current of the board.
//
// Instantaneous current (Iinst), by block averaging: the INA226 gives one
// shunt reading per conversion (~0.34 ms with the defaults). That profiles
// the CNN (~55 ms per inference) but is longer than one MLP or RF
// inference. So the current is reported as averages over blocks:
//   - the stream (STREAM 1): mean current over every BLOCK conversions,
//     as a time series (profile of one inference for slow models);
//   - per burst: charge above idle Q_net = integral of (I - I_idle) dt over
//     the N back-to-back inferences, and the mean current of one inference
//     above idle dI_inf = Q_net / (N * t_inf) (absolute: I_inf = I_idle +
//     dI_inf). The burst holds thousands of conversions even when one
//     inference is shorter than a conversion.
//
// AUTO mode (current/energy test)
// -------------------------------
// The DUT runs the energy firmware (<model>_energy environments): it has no
// serial link during the test, and loops "idle 5 s -> burst of inferences
// 3 s". It toggles its sync pin right before each inference; this logger
// counts both edges in hardware (PCNT), which gives N exactly and tells
// where each burst is:
//   - a burst starts at the first edge after an idle period, and ends when
//     no edge came for `gap` ms (longer than one inference);
//   - its window starts kPreSamples conversions before the first edge and
//     ends at that gap (idle, removed by the subtraction);
//   - the idle baseline of each burst is the idle period right before it,
//     so slow drifts (temperature, supply) cancel burst by burst;
//   - t_inf = (last edge - first edge) / (N - 1).
// One BURST line is printed per burst.
//
// Serial protocol (UART0 at the console baud rate, 921600, see
// sdkconfig.defaults; one command per line, case-insensitive)
// ----------------------------------------------------------------------
//   PING                    -> PONG
//   INFO                    -> INFO,... (current configuration)
//   READ                    -> READ,... (latest conversion + counters)
//   CFG <avg> <vbus_us> <vshunt_us>
//                           INA226 averaging and conversion times, e.g.
//                           CFG 1 140 204 (default)
//   RSHUNT <ohms>           shunt resistor (default 0.1, "R100")
//   BLOCK <n>               conversions per stream point (default 1)
//   STREAM <0|1>            stream Iinst points while a window is open
//   AUTO <bursts> [gap_ms]  measure the next <bursts> bursts of the DUT
//                           (0 = until ABORT), end-of-burst gap (1000 ms)
//   BASELINE [ms]           manual mode: measure the DUT at rest (3000 ms)
//   START                   manual mode: open a run window
//   STOP <n> [t_inf_us]     manual mode: close it; n inferences ran inside
//                           it, t_inf_us is the per-inference time
//   ABORT                   drop the open window / stop AUTO
//
// Answers: OK,<command> / ERR,<reason>. Lines starting with '#' are for
// people only. Data lines (COLUMNS lines list their fields):
//   BURST,index,n_inf,t_inf_us,busy_us,window_us,samples,missed,
//         stream_dropped,idle_us,idle_samples,I_idle_mA,V_idle_V,P_idle_mW,
//         P_idle_std_mW,I_mean_mA,V_mean_V,P_mean_mW,E_total_uJ,E_idle_uJ,
//         E_net_uJ,E_inf_uJ,E_inf_sigma_uJ,dI_inf_mA,dP_inf_mW,I_inf_mA,
//         P_inf_mW
//   AUTO_DONE,bursts
//   BASELINE,window_us,samples,missed,I_mA,V_V,P_mW,P_std_mW,E_uJ
//   RESULT,n_inf,t_inf_us,window_us,samples,missed,stream_dropped,
//          I_mean_mA,V_mean_V,P_mean_mW,E_total_uJ,I_idle_mA,P_idle_mW,
//          E_idle_uJ,E_net_uJ,E_inf_uJ,E_inf_sigma_uJ,dI_window_mA,
//          dI_inf_mA,dP_inf_mW
//   I,window,t_us,I_mA,P_mW   (stream; window = burst index in AUTO, t from
//                             the window start, at the block middle)
// Values that can't be computed (no baseline, n < 2, ...) are "nan".

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/pulse_cnt.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "ina226.h"

namespace {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
constexpr gpio_num_t kSdaPin = GPIO_NUM_21;
constexpr gpio_num_t kSclPin = GPIO_NUM_22;
constexpr i2c_port_num_t kI2cPort = I2C_NUM_0;
// INA226 fast mode. The sampler needs 3 register reads per conversion
// (Mask/Enable, shunt, bus), which fits the default ~344 us conversion.
constexpr uint32_t kI2cClockHz = 400000;
constexpr int kI2cProbeTimeoutMs = 50;

// Sync input from the DUT (toggles before each inference). No pull
// resistor: the DUT drives it, and a pull would load the DUT pin with a
// current that changes with the pin level, inside the measurement.
constexpr gpio_num_t kSyncPin = GPIO_NUM_4;
// Pulses shorter than this are ringing, not inferences (the shortest
// inference is ~100 us).
constexpr uint32_t kSyncGlitchNs = 1000;
constexpr int kPcntHighLimit = 32767;
constexpr int kPcntLowLimit = -32768;

constexpr uart_port_t kUart = UART_NUM_0;
constexpr int kUartRxBufferBytes = 1024;
// Room for the stream: uart_write_bytes() only blocks once it is full.
constexpr int kUartTxBufferBytes = 8192;
// Set here, not only through CONFIG_ESP_CONSOLE_UART_BAUDRATE: a regenerated
// sdkconfig falls back to 115200, too slow for the stream. The boot log
// before this point stays at the sdkconfig rate.
constexpr uint32_t kUartBaudRate = 921600;

constexpr uint8_t kFirstInaAddress = 0x40;
constexpr uint8_t kLastInaAddress = 0x4F;

// "R100" on the usual INA226 modules. With the 2.5 uV shunt LSB this gives
// 25 uA resolution and 819 mA full scale.
constexpr float kDefaultShuntOhms = 0.1f;
// AVG = 1, VBUSCT = 140 us, VSHCT = 204 us: one reading every ~344 us.
constexpr uint16_t kDefaultAveraging = 1;
constexpr uint16_t kDefaultBusTimeUs = 140;
constexpr uint16_t kDefaultShuntTimeUs = 204;
constexpr uint32_t kDefaultBaselineMs = 3000;
constexpr uint32_t kMaxWindowMs = 10u * 60u * 1000u;

// AUTO: end of a burst = no sync edge for this long. Must be longer than
// the slowest inference and shorter than the DUT idle period (5 s).
constexpr uint32_t kDefaultGapMs = 1000;
constexpr uint32_t kMinGapMs = 50;
constexpr uint32_t kMaxGapMs = 4000;
// Conversions before the first edge that go into the burst window (and not
// into the idle baseline): the first inference may start inside the
// conversion before the one where its edge is seen.
constexpr int kPreSamples = 4;
// An idle baseline shorter than this is reported as nan.
constexpr uint32_t kMinIdleSamples = 100;
constexpr size_t kBurstQueueLength = 8;

// A 3 s burst at one conversion per ~344 us is ~9000 points: the queue
// should hold a whole burst at BLOCK 1 while the UART catches up during
// idle. The largest size that fits in one heap block is used; with a
// smaller queue, use BLOCK 2 or more for the profile of long bursts.
constexpr size_t kStreamQueueLengths[] = {10000, 6000, 4000, 2048, 512};
// Stream points printed per pass of the command loop, so commands are still
// read when the stream outruns the UART.
constexpr int kStreamPrintBudget = 64;
constexpr size_t kCommandSize = 128;
// A window whose end passed this long ago without a new conversion is
// closed by the command task (sampler stuck on I2C errors).
constexpr int64_t kCloseTimeoutUs = 200000;

// Sampler alone on core 0; commands and output on core 1.
constexpr int kSamplerCore = 0;
constexpr UBaseType_t kSamplerPriority = 5;
constexpr uint32_t kSamplerStackBytes = 4096;
constexpr int kCommandCore = 1;
constexpr UBaseType_t kCommandPriority = 3;
// vsnprintf() with floats needs more than the 3.5 KB main task stack.
constexpr uint32_t kCommandStackBytes = 8192;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------
struct Sample {
    float current_ma;
    float voltage_v;
    float power_mw;
};

// Time integrals over a window. Each conversion is weighted by the time
// since the previous one (clipped to the window start), so the integrals
// stay right when the sampler misses a conversion or the period drifts.
struct Accumulator {
    double covered_us;
    double charge;     // mA * us
    double energy;     // mW * us (= nJ)
    double volt_time;  // V * us
    // Per-sample sums for the standard deviation (noise floor).
    double power_sum;
    double power_square_sum;
    uint32_t samples;
    uint32_t missed;  // gaps longer than 1.5 nominal conversion periods
};

enum class WindowKind : uint8_t { kNone, kBaseline, kRun };

// Manual mode window (BASELINE, START/STOP).
struct Window {
    WindowKind kind;
    uint32_t id;
    int64_t start_us;
    int64_t end_us;  // 0 = open until STOP
    bool closed;     // closed by the sampler, result not printed yet
    Accumulator accumulator;
};

struct StreamPoint {
    uint32_t window_id;
    uint32_t time_us;  // block middle, from window start
    float current_ma;
    float power_mw;
};

// One AUTO burst, from the sampler to the command task.
struct BurstRecord {
    uint32_t index;
    uint32_t inferences;  // sync edges
    int64_t first_edge_us;
    int64_t last_edge_us;
    uint32_t stream_dropped;
    Accumulator idle;    // idle period right before the burst
    Accumulator window;  // burst + the end-of-burst gap
};

struct ConfigRequest {
    uint8_t averaging_code;
    uint8_t bus_time_code;
    uint8_t shunt_time_code;
};

struct Baseline {
    bool valid;
    double covered_us;
    uint32_t samples;
    double current_ma;
    double voltage_v;
    double power_mw;
    double power_std_mw;
};

Ina226 ina;
i2c_master_bus_handle_t i2c_bus = nullptr;
pcnt_unit_handle_t sync_counter = nullptr;
QueueHandle_t stream_queue = nullptr;
QueueHandle_t burst_queue = nullptr;

// Guards everything the sampler and the command task both touch below.
portMUX_TYPE lock = portMUX_INITIALIZER_UNLOCKED;
Window window = {};
uint32_t next_window_id = 1;
Sample latest = {};
int64_t latest_us = 0;
bool config_pending = false;
bool config_ok = false;
ConfigRequest config_request = {};
// AUTO settings; a new generation makes the sampler restart AUTO.
bool auto_enabled = false;
uint32_t auto_generation = 0;
uint32_t auto_target = 0;  // bursts, 0 = no limit
int64_t auto_gap_us = 0;

// Written by the command task only while no window is open.
std::atomic<float> shunt_ohms{kDefaultShuntOhms};
std::atomic<uint32_t> block_size{1};
std::atomic<bool> stream_enabled{false};
std::atomic<uint32_t> period_us{0};

std::atomic<uint32_t> total_samples{0};
std::atomic<uint32_t> total_missed{0};
std::atomic<uint32_t> i2c_errors{0};
std::atomic<uint32_t> stream_dropped{0};
std::atomic<uint32_t> bursts_dropped{0};

// Command task only.
Baseline baseline = {};
uint32_t run_inferences = 0;
float run_inference_us = 0.0f;

// ---------------------------------------------------------------------------
// Sampler task: the only user of the I2C bus after start-up
// ---------------------------------------------------------------------------
Sample convert(int16_t shunt_raw, uint16_t bus_raw)
{
    Sample sample;
    // V_shunt / R, in mA.
    sample.current_ma = shunt_raw * Ina226::kShuntLsbVolts * 1000.0f /
                        shunt_ohms.load(std::memory_order_relaxed);
    sample.voltage_v = bus_raw * Ina226::kBusLsbVolts;
    sample.power_mw = sample.voltage_v * sample.current_ma;
    return sample;
}

void accumulate(Accumulator &accumulator, const Sample &sample, double dt_us,
                uint32_t missed)
{
    accumulator.covered_us += dt_us;
    accumulator.charge += sample.current_ma * dt_us;
    accumulator.energy += sample.power_mw * dt_us;
    accumulator.volt_time += sample.voltage_v * dt_us;
    accumulator.power_sum += sample.power_mw;
    accumulator.power_square_sum +=
        static_cast<double>(sample.power_mw) * sample.power_mw;
    accumulator.samples += 1;
    accumulator.missed += missed;
}

// Stream block being filled; sampler task only.
struct StreamBlock {
    uint32_t window_id;
    int64_t window_start_us;
    int64_t start_us;
    double covered_us;
    double charge;
    double energy;
    uint32_t count;
};

StreamBlock block = {};

void push_block()
{
    if (block.count == 0 || block.covered_us <= 0.0) {
        return;
    }
    StreamPoint point;
    point.window_id = block.window_id;
    point.time_us = static_cast<uint32_t>(
        block.start_us - block.window_start_us + block.covered_us / 2.0);
    point.current_ma = static_cast<float>(block.charge / block.covered_us);
    point.power_mw = static_cast<float>(block.energy / block.covered_us);
    if (xQueueSend(stream_queue, &point, 0) != pdTRUE) {
        stream_dropped.fetch_add(1, std::memory_order_relaxed);
    }
    block.count = 0;
    block.covered_us = 0.0;
    block.charge = 0.0;
    block.energy = 0.0;
}

void start_block(uint32_t window_id, int64_t window_start_us)
{
    block = {};
    block.window_id = window_id;
    block.window_start_us = window_start_us;
}

void add_to_block(uint32_t window_id, int64_t window_start_us, int64_t from_us,
                  const Sample &sample, double dt_us)
{
    if (block.window_id != window_id) {
        start_block(window_id, window_start_us);
    }
    if (block.count == 0) {
        block.start_us = from_us;
    }
    block.covered_us += dt_us;
    block.charge += sample.current_ma * dt_us;
    block.energy += sample.power_mw * dt_us;
    block.count += 1;
    if (block.count >= block_size.load(std::memory_order_relaxed)) {
        push_block();
    }
}

// Manual windows (BASELINE, START/STOP).
void manual_window_sample(int64_t now_us, int64_t previous_us, const Sample &sample,
                          uint32_t missed)
{
    bool closing = false;
    bool streaming = false;
    uint32_t window_id = 0;
    int64_t window_start_us = 0;
    int64_t from_us = 0;
    double dt_us = 0.0;

    portENTER_CRITICAL(&lock);
    if (window.kind != WindowKind::kNone && !window.closed) {
        window_id = window.id;
        window_start_us = window.start_us;
        if (window.end_us != 0 && now_us > window.end_us) {
            closing = true;
        } else {
            from_us = previous_us > window.start_us ? previous_us : window.start_us;
            if (now_us > from_us) {
                dt_us = static_cast<double>(now_us - from_us);
                accumulate(window.accumulator, sample, dt_us, missed);
                streaming = stream_enabled.load(std::memory_order_relaxed);
            }
        }
    }
    portEXIT_CRITICAL(&lock);

    if (streaming) {
        add_to_block(window_id, window_start_us, from_us, sample, dt_us);
    }
    if (closing) {
        // The last partial block goes in the queue before the command task
        // can see the window closed, so it is printed before the result.
        if (block.window_id == window_id) {
            push_block();
        }
        portENTER_CRITICAL(&lock);
        if (window.id == window_id) {
            window.closed = true;
        }
        portEXIT_CRITICAL(&lock);
    }
}

// AUTO mode state; sampler task only.
enum class AutoPhase : uint8_t { kOff, kSettle, kIdle, kBurst };

struct PendingSample {
    Sample sample;
    int64_t from_us;
    double dt_us;
    uint32_t missed;
};

struct AutoState {
    AutoPhase phase;
    uint32_t generation;
    uint32_t target;
    int64_t gap_us;
    uint32_t bursts;
    int last_count;
    int64_t last_edge_us;
    // Idle: the last kPreSamples conversions wait here before going into
    // the baseline, so the ones right before a burst go into the burst.
    PendingSample pending[kPreSamples];
    int pending_count;
    int pending_head;  // oldest
    Accumulator idle;
    // Burst.
    Accumulator burst;
    int start_count;
    int64_t first_edge_us;
    int64_t window_start_us;
};

AutoState automatic = {};

void reset_idle()
{
    automatic.idle = {};
    automatic.pending_count = 0;
    automatic.pending_head = 0;
}

void add_to_burst(const PendingSample &entry, bool streaming)
{
    accumulate(automatic.burst, entry.sample, entry.dt_us, entry.missed);
    if (streaming) {
        add_to_block(automatic.bursts, automatic.window_start_us, entry.from_us,
                     entry.sample, entry.dt_us);
    }
}

void begin_burst(int64_t now_us, const PendingSample &current, bool streaming)
{
    automatic.bursts += 1;
    automatic.burst = {};
    automatic.start_count = automatic.last_count;
    automatic.first_edge_us = now_us;
    automatic.window_start_us = automatic.pending_count > 0
        ? automatic.pending[automatic.pending_head].from_us
        : current.from_us;
    start_block(automatic.bursts, automatic.window_start_us);
    for (int i = 0; i < automatic.pending_count; ++i) {
        add_to_burst(automatic.pending[(automatic.pending_head + i) % kPreSamples], streaming);
    }
    automatic.pending_count = 0;
    automatic.pending_head = 0;
    add_to_burst(current, streaming);
    automatic.phase = AutoPhase::kBurst;
}

void end_burst(int count)
{
    push_block();

    BurstRecord record;
    record.index = automatic.bursts;
    record.inferences = static_cast<uint32_t>(count - automatic.start_count);
    record.first_edge_us = automatic.first_edge_us;
    record.last_edge_us = automatic.last_edge_us;
    record.stream_dropped = stream_dropped.exchange(0);
    record.idle = automatic.idle;
    record.window = automatic.burst;
    if (xQueueSend(burst_queue, &record, 0) != pdTRUE) {
        bursts_dropped.fetch_add(1, std::memory_order_relaxed);
    }

    reset_idle();
    automatic.phase = AutoPhase::kIdle;
    if (automatic.target != 0 && automatic.bursts >= automatic.target) {
        automatic.phase = AutoPhase::kOff;
        portENTER_CRITICAL(&lock);
        if (auto_generation == automatic.generation) {
            auto_enabled = false;
        }
        portEXIT_CRITICAL(&lock);
    }
}

void auto_sample(int64_t now_us, int64_t previous_us, const Sample &sample,
                 uint32_t missed, int count)
{
    bool enabled;
    uint32_t generation;
    uint32_t target;
    int64_t gap_us;
    portENTER_CRITICAL(&lock);
    enabled = auto_enabled;
    generation = auto_generation;
    target = auto_target;
    gap_us = auto_gap_us;
    portEXIT_CRITICAL(&lock);

    if (generation != automatic.generation) {
        // (Re)armed or aborted: start over. Wait for a quiet sync line
        // first, so a burst already running is not measured from its middle.
        automatic = {};
        automatic.generation = generation;
        automatic.target = target;
        automatic.gap_us = gap_us;
        automatic.phase = enabled ? AutoPhase::kSettle : AutoPhase::kOff;
        automatic.last_count = count;
        automatic.last_edge_us = now_us;
        return;
    }
    if (automatic.phase == AutoPhase::kOff) {
        automatic.last_count = count;
        return;
    }

    const bool edge = count != automatic.last_count;
    if (edge) {
        automatic.last_edge_us = now_us;
    }
    const bool quiet = now_us - automatic.last_edge_us > automatic.gap_us;
    const bool streaming = stream_enabled.load(std::memory_order_relaxed);
    const PendingSample current = {sample, previous_us,
                                   static_cast<double>(now_us - previous_us), missed};

    switch (automatic.phase) {
    case AutoPhase::kSettle:
        if (quiet) {
            reset_idle();
            automatic.phase = AutoPhase::kIdle;
        }
        break;
    case AutoPhase::kIdle:
        if (edge) {
            begin_burst(now_us, current, streaming);
            break;
        }
        if (automatic.pending_count == kPreSamples) {
            const PendingSample &oldest = automatic.pending[automatic.pending_head];
            accumulate(automatic.idle, oldest.sample, oldest.dt_us, oldest.missed);
            automatic.pending[automatic.pending_head] = current;
            automatic.pending_head = (automatic.pending_head + 1) % kPreSamples;
        } else {
            automatic.pending[(automatic.pending_head + automatic.pending_count) %
                              kPreSamples] = current;
            automatic.pending_count += 1;
        }
        break;
    case AutoPhase::kBurst:
        add_to_burst(current, streaming);
        if (quiet) {
            end_burst(count);
        }
        break;
    case AutoPhase::kOff:
        break;
    }
    automatic.last_count = count;
}

void handle_sample(int64_t now_us, int64_t previous_us, const Sample &sample, int count)
{
    const uint32_t nominal_us = period_us.load(std::memory_order_relaxed);
    const int64_t gap_us = now_us - previous_us;
    uint32_t missed = 0;
    if (nominal_us > 0 && gap_us * 2 > static_cast<int64_t>(nominal_us) * 3) {
        missed = static_cast<uint32_t>((gap_us + nominal_us / 2) / nominal_us) - 1;
    }
    total_samples.fetch_add(1, std::memory_order_relaxed);
    total_missed.fetch_add(missed, std::memory_order_relaxed);

    portENTER_CRITICAL(&lock);
    latest = sample;
    latest_us = now_us;
    portEXIT_CRITICAL(&lock);

    manual_window_sample(now_us, previous_us, sample, missed);
    auto_sample(now_us, previous_us, sample, missed, count);
}

void apply_config_request(int64_t &previous_us)
{
    ConfigRequest request;
    bool pending;
    portENTER_CRITICAL(&lock);
    pending = config_pending;
    request = config_request;
    portEXIT_CRITICAL(&lock);
    if (!pending) {
        return;
    }

    const bool ok = ina.configure(request.averaging_code, request.bus_time_code,
                                  request.shunt_time_code) == ESP_OK;
    if (ok) {
        period_us.store(ina.conversion_period_us(), std::memory_order_relaxed);
    }
    // Writing the configuration restarts the conversion.
    previous_us = esp_timer_get_time();

    portENTER_CRITICAL(&lock);
    config_ok = ok;
    config_pending = false;
    portEXIT_CRITICAL(&lock);
}

int sync_edges()
{
    int count = 0;
    pcnt_unit_get_count(sync_counter, &count);
    return count;
}

void sampler_task(void *)
{
    int64_t previous_us = esp_timer_get_time();
    for (;;) {
        apply_config_request(previous_us);

        // Each I2C transfer blocks this task until the bus is done, so the
        // polling leaves the CPU to the idle task in between.
        bool ready = false;
        if (ina.conversion_ready(ready) != ESP_OK) {
            i2c_errors.fetch_add(1, std::memory_order_relaxed);
            vTaskDelay(1);
            continue;
        }
        if (!ready) {
            continue;
        }

        const int64_t now_us = esp_timer_get_time();
        const int count = sync_edges();
        int16_t shunt_raw = 0;
        uint16_t bus_raw = 0;
        if (ina.read_raw(shunt_raw, bus_raw) != ESP_OK) {
            i2c_errors.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        handle_sample(now_us, previous_us, convert(shunt_raw, bus_raw), count);
        previous_us = now_us;
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
void print_line(const char *format, ...) __attribute__((format(printf, 1, 2)));

void print_line(const char *format, ...)
{
    char line[512];
    va_list arguments;
    va_start(arguments, format);
    const int length = std::vsnprintf(line, sizeof(line) - 1, format, arguments);
    va_end(arguments);
    if (length < 0) {
        return;
    }
    size_t size = static_cast<size_t>(length) < sizeof(line) - 1
                      ? static_cast<size_t>(length)
                      : sizeof(line) - 2;
    line[size++] = '\n';
    uart_write_bytes(kUart, line, size);
}

// "%.4f", or "nan" when the value can't be computed.
const char *number(char *buffer, size_t size, double value, int decimals = 4)
{
    if (std::isfinite(value)) {
        std::snprintf(buffer, size, "%.*f", decimals, value);
    } else {
        std::snprintf(buffer, size, "nan");
    }
    return buffer;
}

void print_columns()
{
    print_line("COLUMNS,BURST,index,n_inf,t_inf_us,busy_us,window_us,samples,missed,"
               "stream_dropped,idle_us,idle_samples,I_idle_mA,V_idle_V,P_idle_mW,"
               "P_idle_std_mW,I_mean_mA,V_mean_V,P_mean_mW,E_total_uJ,E_idle_uJ,"
               "E_net_uJ,E_inf_uJ,E_inf_sigma_uJ,dI_inf_mA,dP_inf_mW,I_inf_mA,P_inf_mW");
    print_line("COLUMNS,BASELINE,window_us,samples,missed,I_mA,V_V,P_mW,"
               "P_std_mW,E_uJ");
    print_line("COLUMNS,RESULT,n_inf,t_inf_us,window_us,samples,missed,"
               "stream_dropped,I_mean_mA,V_mean_V,P_mean_mW,E_total_uJ,"
               "I_idle_mA,P_idle_mW,E_idle_uJ,E_net_uJ,E_inf_uJ,"
               "E_inf_sigma_uJ,dI_window_mA,dI_inf_mA,dP_inf_mW");
    print_line("COLUMNS,I,window,t_us,I_mA,P_mW");
}

void print_info()
{
    char ohms[16];
    print_line("INFO,0x%02X,%s,%u,%u,%u,%u,%u,%u,%d,%d,%d", ina.address(),
               number(ohms, sizeof(ohms), shunt_ohms.load(), 4),
               Ina226::kAveragingCounts[ina.averaging_code()],
               Ina226::kConversionTimesUs[ina.bus_time_code()],
               Ina226::kConversionTimesUs[ina.shunt_time_code()],
               static_cast<unsigned>(period_us.load()),
               static_cast<unsigned>(kI2cClockHz),
               static_cast<unsigned>(block_size.load()),
               stream_enabled.load() ? 1 : 0, baseline.valid ? 1 : 0,
               static_cast<int>(kSyncPin));
    print_line("# INFO fields: address,shunt_ohms,averaging,vbus_ct_us,"
               "vshunt_ct_us,period_us,i2c_hz,block,stream,baseline_valid,sync_gpio");
}

void print_stream_point(const StreamPoint &point)
{
    char current[16];
    char power[16];
    print_line("I,%u,%u,%s,%s", static_cast<unsigned>(point.window_id),
               static_cast<unsigned>(point.time_us),
               number(current, sizeof(current), point.current_ma, 3),
               number(power, sizeof(power), point.power_mw, 3));
}

bool print_stream(int budget)
{
    bool printed = false;
    StreamPoint point;
    while (budget != 0 && xQueueReceive(stream_queue, &point, 0) == pdTRUE) {
        print_stream_point(point);
        printed = true;
        if (budget > 0) {
            --budget;
        }
    }
    return printed;
}

double power_std(const Accumulator &accumulator)
{
    if (accumulator.samples < 2) {
        return NAN;
    }
    const double n = accumulator.samples;
    const double mean = accumulator.power_sum / n;
    const double variance = (accumulator.power_square_sum - n * mean * mean) / (n - 1.0);
    return variance > 0.0 ? std::sqrt(variance) : 0.0;
}

// Noise floor of E_net: the uncertainty of both mean powers, with the idle
// sample spread as the noise and samples taken as independent. Only a guide
// for choosing N and the window length.
double net_energy_sigma_uj(double window_us, double idle_power_std_mw,
                           uint32_t idle_samples, uint32_t window_samples)
{
    if (!std::isfinite(idle_power_std_mw) || idle_samples == 0 || window_samples == 0) {
        return NAN;
    }
    return window_us * 1e-3 * idle_power_std_mw *
           std::sqrt(1.0 / idle_samples + 1.0 / window_samples);
}

void print_burst(const BurstRecord &record)
{
    const Accumulator &idle = record.idle;
    const Accumulator &window = record.window;
    const double t = window.covered_us;
    const double n = record.inferences;

    double idle_current = NAN;
    double idle_voltage = NAN;
    double idle_power = NAN;
    double idle_std = NAN;
    if (idle.samples >= kMinIdleSamples && idle.covered_us > 0.0) {
        idle_current = idle.charge / idle.covered_us;
        idle_voltage = idle.volt_time / idle.covered_us;
        idle_power = idle.energy / idle.covered_us;
        idle_std = power_std(idle);
    }
    const double current_mean = t > 0.0 ? window.charge / t : NAN;
    const double voltage_mean = t > 0.0 ? window.volt_time / t : NAN;
    const double power_mean = t > 0.0 ? window.energy / t : NAN;
    const double energy_total_uj = window.energy * 1e-3;
    const double idle_energy_uj = idle_power * t * 1e-3;
    const double net_energy_uj = energy_total_uj - idle_energy_uj;
    const double net_charge = window.charge - idle_current * t;  // mA * us

    // Edges are at inference starts: N - 1 inferences between the first
    // and the last one.
    const double inference_us = n >= 2
        ? static_cast<double>(record.last_edge_us - record.first_edge_us) / (n - 1.0)
        : NAN;
    const double busy_us = n * inference_us;
    const double energy_inference_uj = n > 0 ? net_energy_uj / n : NAN;
    const double sigma_uj = n > 0
        ? net_energy_sigma_uj(t, idle_std, idle.samples, window.samples) / n
        : NAN;
    const double delta_current = net_charge / busy_us;
    const double delta_power = net_energy_uj * 1e3 / busy_us;

    char c[22][24];
    print_line("BURST,%u,%u,%s,%s,%.0f,%u,%u,%u,%.0f,%u,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
               "%s,%s,%s,%s",
               static_cast<unsigned>(record.index), static_cast<unsigned>(record.inferences),
               number(c[0], sizeof(c[0]), inference_us, 2),
               number(c[1], sizeof(c[1]), busy_us, 0),
               t, static_cast<unsigned>(window.samples),
               static_cast<unsigned>(window.missed + idle.missed),
               static_cast<unsigned>(record.stream_dropped), idle.covered_us,
               static_cast<unsigned>(idle.samples),
               number(c[2], sizeof(c[2]), idle_current),
               number(c[3], sizeof(c[3]), idle_voltage),
               number(c[4], sizeof(c[4]), idle_power),
               number(c[5], sizeof(c[5]), idle_std),
               number(c[6], sizeof(c[6]), current_mean),
               number(c[7], sizeof(c[7]), voltage_mean),
               number(c[8], sizeof(c[8]), power_mean),
               number(c[9], sizeof(c[9]), energy_total_uj),
               number(c[10], sizeof(c[10]), idle_energy_uj),
               number(c[11], sizeof(c[11]), net_energy_uj),
               number(c[12], sizeof(c[12]), energy_inference_uj),
               number(c[13], sizeof(c[13]), sigma_uj),
               number(c[14], sizeof(c[14]), delta_current),
               number(c[15], sizeof(c[15]), delta_power),
               number(c[16], sizeof(c[16]), idle_current + delta_current),
               number(c[17], sizeof(c[17]), idle_power + delta_power));
    print_line("# Burst %u: %u inferences of %.1f us, Einf %.4f uJ (+- %.4f), "
               "I %.3f mA idle -> %.3f mA inferring",
               static_cast<unsigned>(record.index), static_cast<unsigned>(record.inferences),
               inference_us, energy_inference_uj, sigma_uj, idle_current,
               idle_current + delta_current);
}

void finish_baseline(const Accumulator &accumulator)
{
    if (accumulator.samples == 0 || accumulator.covered_us <= 0.0) {
        print_line("ERR,baseline window got no samples (check the INA226)");
        return;
    }
    const double t = accumulator.covered_us;
    baseline.valid = true;
    baseline.covered_us = t;
    baseline.samples = accumulator.samples;
    baseline.current_ma = accumulator.charge / t;
    baseline.voltage_v = accumulator.volt_time / t;
    baseline.power_mw = accumulator.energy / t;
    baseline.power_std_mw = power_std(accumulator);

    char c[5][24];
    print_line("BASELINE,%.0f,%u,%u,%s,%s,%s,%s,%s", t,
               static_cast<unsigned>(accumulator.samples),
               static_cast<unsigned>(accumulator.missed),
               number(c[0], sizeof(c[0]), baseline.current_ma),
               number(c[1], sizeof(c[1]), baseline.voltage_v),
               number(c[2], sizeof(c[2]), baseline.power_mw),
               number(c[3], sizeof(c[3]), baseline.power_std_mw),
               number(c[4], sizeof(c[4]), accumulator.energy * 1e-3));
    print_line("# Idle: %.3f mA, %.4f V, %.3f mW over %.2f s (%u samples)",
               baseline.current_ma, baseline.voltage_v, baseline.power_mw,
               t * 1e-6, static_cast<unsigned>(accumulator.samples));
}

void finish_run(const Accumulator &accumulator, uint32_t inferences,
                float inference_us, uint32_t dropped)
{
    if (accumulator.samples == 0 || accumulator.covered_us <= 0.0) {
        print_line("ERR,run window got no samples (check the INA226)");
        return;
    }
    const double t = accumulator.covered_us;
    const double n = inferences;
    const double current_mean = accumulator.charge / t;
    const double voltage_mean = accumulator.volt_time / t;
    const double power_mean = accumulator.energy / t;
    const double energy_total_uj = accumulator.energy * 1e-3;

    double idle_current = NAN;
    double idle_power = NAN;
    double idle_energy_uj = NAN;
    double net_energy_uj = NAN;
    double net_charge = NAN;  // mA * us
    double sigma_uj = NAN;
    if (baseline.valid) {
        idle_current = baseline.current_ma;
        idle_power = baseline.power_mw;
        idle_energy_uj = idle_power * t * 1e-3;
        net_energy_uj = energy_total_uj - idle_energy_uj;
        net_charge = accumulator.charge - idle_current * t;
        sigma_uj = net_energy_sigma_uj(t, baseline.power_std_mw, baseline.samples,
                                       accumulator.samples);
    }

    const double energy_inference_uj = n > 0 ? net_energy_uj / n : NAN;
    const double sigma_inference_uj = n > 0 ? sigma_uj / n : NAN;
    const double delta_window_ma = net_charge / t;
    // The inferences' own share of the window: N * t_inf.
    const double busy_us = n * inference_us;
    const double delta_inference_ma = busy_us > 0 ? net_charge / busy_us : NAN;
    const double delta_power_mw = busy_us > 0 ? net_energy_uj * 1e3 / busy_us : NAN;

    char c[14][24];
    print_line("RESULT,%u,%s,%.0f,%u,%u,%u,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
               static_cast<unsigned>(inferences),
               number(c[0], sizeof(c[0]), inference_us > 0 ? inference_us : NAN, 1),
               t, static_cast<unsigned>(accumulator.samples),
               static_cast<unsigned>(accumulator.missed),
               static_cast<unsigned>(dropped),
               number(c[1], sizeof(c[1]), current_mean),
               number(c[2], sizeof(c[2]), voltage_mean),
               number(c[3], sizeof(c[3]), power_mean),
               number(c[4], sizeof(c[4]), energy_total_uj),
               number(c[5], sizeof(c[5]), idle_current),
               number(c[6], sizeof(c[6]), idle_power),
               number(c[7], sizeof(c[7]), idle_energy_uj),
               number(c[8], sizeof(c[8]), net_energy_uj),
               number(c[9], sizeof(c[9]), energy_inference_uj),
               number(c[10], sizeof(c[10]), sigma_inference_uj),
               number(c[11], sizeof(c[11]), delta_window_ma),
               number(c[12], sizeof(c[12]), delta_inference_ma),
               number(c[13], sizeof(c[13]), delta_power_mw));

    if (!baseline.valid) {
        print_line("# No baseline: run BASELINE first for the net values.");
        return;
    }
    print_line("# Window %.3f s: %.3f mA / %.3f mW mean, idle %.3f mA / %.3f mW",
               t * 1e-6, current_mean, power_mean, idle_current, idle_power);
    print_line("# E_net %.3f uJ over %u inferences -> Einf %.4f uJ (+- %.4f)",
               net_energy_uj, static_cast<unsigned>(inferences),
               energy_inference_uj, sigma_inference_uj);
    if (busy_us > 0) {
        print_line("# Per inference above idle: dI %.3f mA, dP %.3f mW "
                   "(DUT busy %.1f%% of the window)",
                   delta_inference_ma, delta_power_mw, 100.0 * busy_us / t);
    }
    if (accumulator.missed > 0 || dropped > 0) {
        print_line("# Warning: %u conversions missed, %u stream points dropped",
                   static_cast<unsigned>(accumulator.missed),
                   static_cast<unsigned>(dropped));
    }
}

// Prints the result once the sampler has closed a manual window.
bool check_window()
{
    Window snapshot;
    bool closed;
    const int64_t now_us = esp_timer_get_time();
    portENTER_CRITICAL(&lock);
    if (window.kind != WindowKind::kNone && !window.closed && window.end_us != 0 &&
        now_us > window.end_us + kCloseTimeoutUs) {
        window.closed = true;  // sampler stuck, see kCloseTimeoutUs
    }
    closed = window.kind != WindowKind::kNone && window.closed;
    snapshot = window;
    if (closed) {
        window.kind = WindowKind::kNone;
        window.closed = false;
    }
    portEXIT_CRITICAL(&lock);
    if (!closed) {
        return false;
    }

    // Stream points of this window come before its result.
    print_stream(-1);
    if (snapshot.kind == WindowKind::kBaseline) {
        finish_baseline(snapshot.accumulator);
    } else {
        finish_run(snapshot.accumulator, run_inferences, run_inference_us,
                   stream_dropped.load());
    }
    return true;
}

// Prints the AUTO bursts the sampler finished.
bool check_bursts()
{
    BurstRecord record;
    if (xQueueReceive(burst_queue, &record, 0) != pdTRUE) {
        return false;
    }
    // Stream points of this burst come before its result.
    print_stream(-1);
    print_burst(record);
    const uint32_t lost = bursts_dropped.exchange(0);
    if (lost > 0) {
        print_line("# Warning: %u burst results lost (output too slow)",
                   static_cast<unsigned>(lost));
    }

    uint32_t target;
    portENTER_CRITICAL(&lock);
    target = auto_target;
    portEXIT_CRITICAL(&lock);
    if (target != 0 && record.index >= target) {
        print_line("AUTO_DONE,%u", static_cast<unsigned>(record.index));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
bool busy()
{
    portENTER_CRITICAL(&lock);
    const bool open = window.kind != WindowKind::kNone || auto_enabled;
    portEXIT_CRITICAL(&lock);
    return open;
}

void open_window(WindowKind kind, uint32_t duration_ms)
{
    // Leftovers from an aborted window.
    xQueueReset(stream_queue);
    stream_dropped.store(0);

    const int64_t now_us = esp_timer_get_time();
    portENTER_CRITICAL(&lock);
    window = {};
    window.kind = kind;
    window.id = next_window_id++;
    window.start_us = now_us;
    window.end_us = duration_ms > 0 ? now_us + static_cast<int64_t>(duration_ms) * 1000 : 0;
    portEXIT_CRITICAL(&lock);
}

bool parse_uint(const char *token, uint32_t &value)
{
    if (token == nullptr) {
        return false;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(token, &end, 10);
    if (end == token || *end != '\0') {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_float(const char *token, float &value)
{
    if (token == nullptr) {
        return false;
    }
    char *end = nullptr;
    value = std::strtof(token, &end);
    return end != token && *end == '\0' && std::isfinite(value);
}

void command_config(char *arguments[], int count)
{
    uint32_t averaging = 0;
    uint32_t bus_us = 0;
    uint32_t shunt_us = 0;
    if (count != 3 || !parse_uint(arguments[0], averaging) ||
        !parse_uint(arguments[1], bus_us) || !parse_uint(arguments[2], shunt_us)) {
        print_line("ERR,usage: CFG <avg> <vbus_ct_us> <vshunt_ct_us>");
        return;
    }
    const int averaging_code = Ina226::code_for(Ina226::kAveragingCounts, averaging);
    const int bus_code = Ina226::code_for(Ina226::kConversionTimesUs, bus_us);
    const int shunt_code = Ina226::code_for(Ina226::kConversionTimesUs, shunt_us);
    if (averaging_code < 0 || bus_code < 0 || shunt_code < 0) {
        print_line("ERR,avg in 1,4,16,64,128,256,512,1024; conversion times in "
                   "140,204,332,588,1100,2116,4156,8244 us");
        return;
    }

    portENTER_CRITICAL(&lock);
    config_request.averaging_code = static_cast<uint8_t>(averaging_code);
    config_request.bus_time_code = static_cast<uint8_t>(bus_code);
    config_request.shunt_time_code = static_cast<uint8_t>(shunt_code);
    config_pending = true;
    portEXIT_CRITICAL(&lock);

    // The sampler owns the bus: wait for it to apply the request.
    const int64_t deadline_us = esp_timer_get_time() + 500000;
    bool pending = true;
    bool ok = false;
    while (pending && esp_timer_get_time() < deadline_us) {
        vTaskDelay(1);
        portENTER_CRITICAL(&lock);
        pending = config_pending;
        ok = config_ok;
        portEXIT_CRITICAL(&lock);
    }
    if (pending || !ok) {
        print_line("ERR,INA226 did not take the configuration");
        return;
    }
    // Baseline and run must use the same INA226 setup.
    baseline.valid = false;
    print_line("OK,CFG");
    print_info();
}

void command_auto(char *arguments[], int count)
{
    uint32_t bursts = 0;
    uint32_t gap_ms = kDefaultGapMs;
    if (count < 1 || count > 2 || !parse_uint(arguments[0], bursts) ||
        (count == 2 && !parse_uint(arguments[1], gap_ms)) ||
        gap_ms < kMinGapMs || gap_ms > kMaxGapMs) {
        print_line("ERR,usage: AUTO <bursts, 0 = until ABORT> [gap_ms, %u..%u]",
                   static_cast<unsigned>(kMinGapMs), static_cast<unsigned>(kMaxGapMs));
        return;
    }
    xQueueReset(stream_queue);
    xQueueReset(burst_queue);
    stream_dropped.store(0);
    bursts_dropped.store(0);
    portENTER_CRITICAL(&lock);
    auto_enabled = true;
    auto_generation += 1;
    auto_target = bursts;
    auto_gap_us = static_cast<int64_t>(gap_ms) * 1000;
    portEXIT_CRITICAL(&lock);
    print_line("OK,AUTO");
    print_line("# Waiting for the DUT bursts (sync on GPIO%d, %d edges so far)",
               static_cast<int>(kSyncPin), sync_edges());
}

void handle_command(char *line)
{
    constexpr int kMaxArguments = 4;
    char *arguments[kMaxArguments] = {};
    int count = 0;
    char *name = std::strtok(line, " \t\r\n,");
    if (name == nullptr) {
        return;
    }
    for (char *token = std::strtok(nullptr, " \t\r\n,");
         token != nullptr && count < kMaxArguments;
         token = std::strtok(nullptr, " \t\r\n,")) {
        arguments[count++] = token;
    }

    const bool measuring = busy();
    const auto is = [name](const char *command) { return strcasecmp(name, command) == 0; };
    const auto refuse_if_busy = [measuring]() {
        if (measuring) {
            print_line("ERR,a measurement is running (STOP or ABORT it first)");
        }
        return measuring;
    };

    if (is("PING")) {
        print_line("PONG");
    } else if (is("INFO")) {
        print_info();
        print_columns();
    } else if (is("READ")) {
        Sample sample;
        int64_t sample_us;
        portENTER_CRITICAL(&lock);
        sample = latest;
        sample_us = latest_us;
        portEXIT_CRITICAL(&lock);
        print_line("READ,%.3f,%.4f,%.3f,%lld,%u,%u,%u,%d", sample.current_ma,
                   sample.voltage_v, sample.power_mw,
                   static_cast<long long>(esp_timer_get_time() - sample_us),
                   static_cast<unsigned>(total_samples.load()),
                   static_cast<unsigned>(total_missed.load()),
                   static_cast<unsigned>(i2c_errors.load()), sync_edges());
        print_line("# READ fields: I_mA,V_V,P_mW,age_us,samples,missed,i2c_errors,sync_edges");
    } else if (is("CFG")) {
        if (!refuse_if_busy()) {
            command_config(arguments, count);
        }
    } else if (is("RSHUNT")) {
        float ohms = 0.0f;
        if (refuse_if_busy()) {
            return;
        }
        if (count != 1 || !parse_float(arguments[0], ohms) || ohms <= 0.0f) {
            print_line("ERR,usage: RSHUNT <ohms>");
            return;
        }
        shunt_ohms.store(ohms);
        baseline.valid = false;
        print_line("OK,RSHUNT");
    } else if (is("BLOCK")) {
        uint32_t size = 0;
        if (refuse_if_busy()) {
            return;
        }
        if (count != 1 || !parse_uint(arguments[0], size) || size == 0 || size > 100000) {
            print_line("ERR,usage: BLOCK <conversions per stream point, 1..100000>");
            return;
        }
        block_size.store(size);
        print_line("OK,BLOCK");
    } else if (is("STREAM")) {
        uint32_t enabled = 0;
        if (refuse_if_busy()) {
            return;
        }
        if (count != 1 || !parse_uint(arguments[0], enabled) || enabled > 1) {
            print_line("ERR,usage: STREAM <0|1>");
            return;
        }
        stream_enabled.store(enabled == 1);
        print_line("OK,STREAM");
    } else if (is("AUTO")) {
        if (!refuse_if_busy()) {
            command_auto(arguments, count);
        }
    } else if (is("BASELINE")) {
        uint32_t duration_ms = kDefaultBaselineMs;
        if (refuse_if_busy()) {
            return;
        }
        if ((count == 1 && !parse_uint(arguments[0], duration_ms)) || count > 1 ||
            duration_ms == 0 || duration_ms > kMaxWindowMs) {
            print_line("ERR,usage: BASELINE [ms], up to %u ms",
                       static_cast<unsigned>(kMaxWindowMs));
            return;
        }
        open_window(WindowKind::kBaseline, duration_ms);
        print_line("OK,BASELINE");
    } else if (is("START")) {
        if (refuse_if_busy()) {
            return;
        }
        run_inferences = 0;
        run_inference_us = 0.0f;
        open_window(WindowKind::kRun, 0);
        print_line("OK,START");
    } else if (is("STOP")) {
        uint32_t inferences = 0;
        float inference_us = 0.0f;
        if (count < 1 || count > 2 || !parse_uint(arguments[0], inferences) ||
            (count == 2 && (!parse_float(arguments[1], inference_us) || inference_us < 0.0f))) {
            print_line("ERR,usage: STOP <n_inferences> [t_inf_us]");
            return;
        }
        bool stopped = false;
        portENTER_CRITICAL(&lock);
        if (window.kind == WindowKind::kRun && !window.closed && window.end_us == 0) {
            // The sampler closes the window at its next conversion.
            window.end_us = esp_timer_get_time();
            stopped = true;
        }
        portEXIT_CRITICAL(&lock);
        if (!stopped) {
            print_line("ERR,no run window open (START first)");
            return;
        }
        run_inferences = inferences;
        run_inference_us = inference_us;
        print_line("OK,STOP");
    } else if (is("ABORT")) {
        portENTER_CRITICAL(&lock);
        window.kind = WindowKind::kNone;
        window.closed = false;
        window.id = next_window_id++;  // the sampler drops its block
        auto_enabled = false;
        auto_generation += 1;
        portEXIT_CRITICAL(&lock);
        xQueueReset(stream_queue);
        xQueueReset(burst_queue);
        print_line("OK,ABORT");
    } else {
        print_line("ERR,unknown command: %s", name);
    }
}

bool read_commands()
{
    static char command[kCommandSize];
    static size_t length = 0;
    static bool overflow = false;
    bool handled = false;
    char character;
    while (uart_read_bytes(kUart, &character, 1, 0) == 1) {
        if (character == '\n' || character == '\r') {
            if (overflow) {
                print_line("ERR,command too long");
            } else if (length > 0) {
                command[length] = '\0';
                handle_command(command);
                handled = true;
            }
            length = 0;
            overflow = false;
        } else if (length < kCommandSize - 1) {
            command[length++] = character;
        } else {
            overflow = true;
        }
    }
    return handled;
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------
void init_console()
{
    uart_driver_install(kUart, kUartRxBufferBytes, kUartTxBufferBytes, 0, nullptr, 0);
    uart_set_baudrate(kUart, kUartBaudRate);
    uart_vfs_dev_use_driver(kUart);
}

bool init_i2c()
{
    i2c_master_bus_config_t config = {};
    config.i2c_port = kI2cPort;
    config.sda_io_num = kSdaPin;
    config.scl_io_num = kSclPin;
    config.clk_source = I2C_CLK_SRC_DEFAULT;
    config.glitch_ignore_cnt = 7;
    // Weak (~45 kOhm); the INA226 modules have their own 10 kOhm pull-ups.
    config.flags.enable_internal_pullup = true;
    return i2c_new_master_bus(&config, &i2c_bus) == ESP_OK;
}

// Hardware counter of the DUT sync edges (both directions).
bool init_sync_counter()
{
    pcnt_unit_config_t unit_config = {};
    unit_config.low_limit = kPcntLowLimit;
    unit_config.high_limit = kPcntHighLimit;
    // Keeps counting past the 16-bit hardware limit (a 3 s RF burst on the
    // STM32 can be tens of thousands of inferences).
    unit_config.flags.accum_count = 1;
    if (pcnt_new_unit(&unit_config, &sync_counter) != ESP_OK) {
        return false;
    }

    pcnt_glitch_filter_config_t filter = {};
    filter.max_glitch_ns = kSyncGlitchNs;
    pcnt_chan_config_t channel_config = {};
    channel_config.edge_gpio_num = kSyncPin;
    channel_config.level_gpio_num = -1;
    pcnt_channel_handle_t channel = nullptr;
    if (pcnt_unit_set_glitch_filter(sync_counter, &filter) != ESP_OK ||
        pcnt_new_channel(sync_counter, &channel_config, &channel) != ESP_OK ||
        pcnt_channel_set_edge_action(channel, PCNT_CHANNEL_EDGE_ACTION_INCREASE,
                                     PCNT_CHANNEL_EDGE_ACTION_INCREASE) != ESP_OK ||
        pcnt_channel_set_level_action(channel, PCNT_CHANNEL_LEVEL_ACTION_KEEP,
                                      PCNT_CHANNEL_LEVEL_ACTION_KEEP) != ESP_OK ||
        pcnt_unit_add_watch_point(sync_counter, kPcntHighLimit) != ESP_OK) {
        return false;
    }
    // pcnt_new_channel() turns the input pull-up on; see kSyncPin.
    gpio_pullup_dis(kSyncPin);
    gpio_pulldown_dis(kSyncPin);
    return pcnt_unit_enable(sync_counter) == ESP_OK &&
           pcnt_unit_clear_count(sync_counter) == ESP_OK &&
           pcnt_unit_start(sync_counter) == ESP_OK;
}

// An idle I2C bus is high on both lines (pull-ups). A line held low means
// the INA226 module is unpowered (its pull-ups go to its VCC), a wire is on
// the wrong pin or shorted to GND: every transfer would just time out.
bool bus_lines_idle()
{
    const int sda = gpio_get_level(kSdaPin);
    const int scl = gpio_get_level(kSclPin);
    if (sda == 1 && scl == 1) {
        return true;
    }
    print_line("ERR,I2C bus held low (SDA=GPIO%d is %d, SCL=GPIO%d is %d): check the INA226 "
               "VCC (3V3) and GND, and that SDA/SCL go to GPIO%d/GPIO%d",
               static_cast<int>(kSdaPin), sda, static_cast<int>(kSclPin), scl,
               static_cast<int>(kSdaPin), static_cast<int>(kSclPin));
    return false;
}

// Lists the devices that answer. False if the bus stops working mid-scan
// (a timeout, not a missing device), so the caller doesn't wait for 126
// timeouts.
bool scan_bus()
{
    int found = 0;
    for (uint16_t address = 1; address < 127; ++address) {
        const esp_err_t result = i2c_master_probe(i2c_bus, address, kI2cProbeTimeoutMs);
        if (result == ESP_OK) {
            print_line("# I2C device at 0x%02X", address);
            ++found;
        } else if (result != ESP_ERR_NOT_FOUND) {
            print_line("ERR,I2C bus timeout at 0x%02X (%s): check the INA226 VCC/GND and "
                       "SDA=GPIO%d, SCL=GPIO%d", address, esp_err_to_name(result),
                       static_cast<int>(kSdaPin), static_cast<int>(kSclPin));
            return false;
        }
    }
    if (found == 0) {
        print_line("# I2C scan: no device (check SDA=GPIO%d, SCL=GPIO%d, 3V3, GND)",
                   static_cast<int>(kSdaPin), static_cast<int>(kSclPin));
    }
    return true;
}

// First INA226 on 0x40..0x4F (A0/A1 straps); retries until one answers.
void find_ina226()
{
    for (;;) {
        if (!bus_lines_idle() || !scan_bus()) {
            print_line("ERR,no INA226 found, retrying in 1 s");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        for (uint8_t address = kFirstInaAddress; address <= kLastInaAddress; ++address) {
            if (i2c_master_probe(i2c_bus, address, kI2cProbeTimeoutMs) == ESP_OK &&
                ina.begin(i2c_bus, address, kI2cClockHz) == ESP_OK) {
                print_line("# INA226 at 0x%02X", address);
                return;
            }
        }
        print_line("ERR,no INA226 found, retrying in 1 s");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void command_task(void *)
{
    print_line("# TinyML power logger (INA226)");
    if (!init_i2c()) {
        print_line("ERR,I2C bus initialization failed");
        vTaskDelete(nullptr);
        return;
    }
    if (!init_sync_counter()) {
        print_line("ERR,sync counter (PCNT) initialization failed");
        vTaskDelete(nullptr);
        return;
    }
    find_ina226();

    const esp_err_t configured = ina.configure(
        static_cast<uint8_t>(Ina226::code_for(Ina226::kAveragingCounts, kDefaultAveraging)),
        static_cast<uint8_t>(Ina226::code_for(Ina226::kConversionTimesUs, kDefaultBusTimeUs)),
        static_cast<uint8_t>(Ina226::code_for(Ina226::kConversionTimesUs, kDefaultShuntTimeUs)));
    if (configured != ESP_OK) {
        print_line("ERR,INA226 configuration failed");
    }
    period_us.store(ina.conversion_period_us());

    burst_queue = xQueueCreate(kBurstQueueLength, sizeof(BurstRecord));
    for (size_t length : kStreamQueueLengths) {
        stream_queue = xQueueCreate(length, sizeof(StreamPoint));
        if (stream_queue != nullptr) {
            print_line("# Stream queue: %u points", static_cast<unsigned>(length));
            break;
        }
    }
    if (stream_queue == nullptr || burst_queue == nullptr) {
        print_line("ERR,out of memory for the output queues");
        vTaskDelete(nullptr);
        return;
    }
    xTaskCreatePinnedToCore(sampler_task, "ina226_sampler", kSamplerStackBytes,
                            nullptr, kSamplerPriority, nullptr, kSamplerCore);

    print_info();
    print_columns();
    print_line("READY");

    for (;;) {
        bool active = read_commands();
        active |= print_stream(kStreamPrintBudget);
        active |= check_window();
        active |= check_bursts();
        if (!active) {
            vTaskDelay(1);
        }
    }
}

}  // namespace

extern "C" void power_logger_main(void)
{
    init_console();
    xTaskCreatePinnedToCore(command_task, "power_logger", kCommandStackBytes,
                            nullptr, kCommandPriority, nullptr, kCommandCore);
}
