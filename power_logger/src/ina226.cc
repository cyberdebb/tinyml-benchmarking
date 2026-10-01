#include "ina226.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

constexpr uint16_t Ina226::kAveragingCounts[];
constexpr uint16_t Ina226::kConversionTimesUs[];

namespace {

// Several ticks: with CONFIG_FREERTOS_HZ=100 a 10 ms timeout is one tick,
// which can run out right after the transfer starts.
constexpr int kTransferTimeoutMs = 50;
constexpr uint16_t kConfigReset = 1u << 15;
// Bits 14..12 of the configuration register read back as 0b100.
constexpr uint16_t kConfigFixedBits = 0x4000;
constexpr uint16_t kModeShuntBusContinuous = 0b111;

}  // namespace

esp_err_t Ina226::begin(i2c_master_bus_handle_t bus, uint8_t address,
                        uint32_t clock_hz)
{
    i2c_device_config_t config = {};
    config.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    config.device_address = address;
    config.scl_speed_hz = clock_hz;
    esp_err_t error = i2c_master_bus_add_device(bus, &config, &device_);
    if (error != ESP_OK) {
        return error;
    }
    address_ = address;

    uint16_t manufacturer = 0;
    uint16_t die = 0;
    error = read_register(kRegManufacturerId, manufacturer);
    if (error == ESP_OK) {
        error = read_register(kRegDieId, die);
    }
    if (error == ESP_OK &&
        (manufacturer != kManufacturerId || (die & kDieIdMask) != kDieId)) {
        error = ESP_ERR_NOT_FOUND;
    }
    if (error == ESP_OK) {
        error = write_register(kRegConfig, kConfigReset);
    }
    if (error != ESP_OK) {
        i2c_master_bus_rm_device(device_);
        device_ = nullptr;
        return error;
    }
    vTaskDelay(pdMS_TO_TICKS(2) + 1);
    return ESP_OK;
}

esp_err_t Ina226::configure(uint8_t averaging_code, uint8_t bus_time_code,
                            uint8_t shunt_time_code)
{
    if (averaging_code >= kCodeCount || bus_time_code >= kCodeCount ||
        shunt_time_code >= kCodeCount) {
        return ESP_ERR_INVALID_ARG;
    }
    const uint16_t config = kConfigFixedBits |
                            static_cast<uint16_t>(averaging_code << 9) |
                            static_cast<uint16_t>(bus_time_code << 6) |
                            static_cast<uint16_t>(shunt_time_code << 3) |
                            kModeShuntBusContinuous;
    const esp_err_t error = write_register(kRegConfig, config);
    if (error != ESP_OK) {
        return error;
    }
    averaging_code_ = averaging_code;
    bus_time_code_ = bus_time_code;
    shunt_time_code_ = shunt_time_code;
    return ESP_OK;
}

esp_err_t Ina226::conversion_ready(bool &ready)
{
    uint16_t mask_enable = 0;
    const esp_err_t error = read_register(kRegMaskEnable, mask_enable);
    ready = error == ESP_OK && (mask_enable & kConversionReadyFlag) != 0;
    return error;
}

esp_err_t Ina226::read_raw(int16_t &shunt, uint16_t &bus)
{
    uint16_t shunt_register = 0;
    esp_err_t error = read_register(kRegShuntVoltage, shunt_register);
    if (error == ESP_OK) {
        error = read_register(kRegBusVoltage, bus);
    }
    // Two's complement, LSB 2.5 uV.
    shunt = static_cast<int16_t>(shunt_register);
    return error;
}

uint32_t Ina226::conversion_period_us() const
{
    return static_cast<uint32_t>(kAveragingCounts[averaging_code_]) *
           (kConversionTimesUs[bus_time_code_] + kConversionTimesUs[shunt_time_code_]);
}

int Ina226::code_for(const uint16_t *table, uint32_t value)
{
    for (int code = 0; code < kCodeCount; ++code) {
        if (table[code] == value) {
            return code;
        }
    }
    return -1;
}

esp_err_t Ina226::read_register(uint8_t reg, uint16_t &value)
{
    // Pointer write + repeated start + 2-byte read, MSB first.
    uint8_t data[2] = {};
    const esp_err_t error = i2c_master_transmit_receive(
        device_, &reg, 1, data, sizeof(data), kTransferTimeoutMs);
    value = static_cast<uint16_t>((data[0] << 8) | data[1]);
    return error;
}

esp_err_t Ina226::write_register(uint8_t reg, uint16_t value)
{
    const uint8_t data[3] = {reg, static_cast<uint8_t>(value >> 8),
                             static_cast<uint8_t>(value & 0xFF)};
    return i2c_master_transmit(device_, data, sizeof(data), kTransferTimeoutMs);
}
