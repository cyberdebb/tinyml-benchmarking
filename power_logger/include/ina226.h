#pragma once

#include <cstdint>

#include "driver/i2c_master.h"
#include "esp_err.h"

// Minimal INA226 driver. Only what the power logger needs: the shunt and
// bus voltage registers are read directly and converted on the ESP32, so
// the CALIBRATION/CURRENT/POWER registers (and their rounding) are not used.
class Ina226 {
public:
    static constexpr uint8_t kRegConfig = 0x00;
    static constexpr uint8_t kRegShuntVoltage = 0x01;
    static constexpr uint8_t kRegBusVoltage = 0x02;
    static constexpr uint8_t kRegMaskEnable = 0x06;
    static constexpr uint8_t kRegManufacturerId = 0xFE;
    static constexpr uint8_t kRegDieId = 0xFF;

    static constexpr uint16_t kManufacturerId = 0x5449;  // "TI"
    static constexpr uint16_t kDieIdMask = 0xFFF0;       // revision dropped
    static constexpr uint16_t kDieId = 0x2260;

    static constexpr float kShuntLsbVolts = 2.5e-6f;
    static constexpr float kBusLsbVolts = 1.25e-3f;

    // Mask/Enable bit 3 (CVRF): set when a shunt + bus conversion (with
    // all of its hardware averages) is done; cleared by reading the
    // Mask/Enable register. ALERT is not wired, so this is polled.
    static constexpr uint16_t kConversionReadyFlag = 1u << 3;

    // Averaging count and conversion time codes (datasheet tables 6 to 8).
    static constexpr int kCodeCount = 8;
    static constexpr uint16_t kAveragingCounts[kCodeCount] = {
        1, 4, 16, 64, 128, 256, 512, 1024};
    static constexpr uint16_t kConversionTimesUs[kCodeCount] = {
        140, 204, 332, 588, 1100, 2116, 4156, 8244};

    // Adds the device at `address` to the bus and checks its IDs; on
    // failure the device is removed again.
    esp_err_t begin(i2c_master_bus_handle_t bus, uint8_t address,
                    uint32_t clock_hz);

    // Continuous shunt + bus mode.
    esp_err_t configure(uint8_t averaging_code, uint8_t bus_time_code,
                        uint8_t shunt_time_code);
    esp_err_t conversion_ready(bool &ready);
    esp_err_t read_raw(int16_t &shunt, uint16_t &bus);

    uint8_t address() const { return address_; }
    uint8_t averaging_code() const { return averaging_code_; }
    uint8_t bus_time_code() const { return bus_time_code_; }
    uint8_t shunt_time_code() const { return shunt_time_code_; }
    // Nominal time between two CVRF (datasheet value, +-10% on the part).
    uint32_t conversion_period_us() const;

    // Index of `value` in one of the tables above, -1 if absent.
    static int code_for(const uint16_t *table, uint32_t value);

private:
    esp_err_t read_register(uint8_t reg, uint16_t &value);
    esp_err_t write_register(uint8_t reg, uint16_t value);

    i2c_master_dev_handle_t device_ = nullptr;
    uint8_t address_ = 0;
    uint8_t averaging_code_ = 0;
    uint8_t bus_time_code_ = 0;
    uint8_t shunt_time_code_ = 0;
};
