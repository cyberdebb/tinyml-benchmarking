#include <stdio.h>

#include "main.h"

extern void tinyml_app_main(void);

UART_HandleTypeDef huart3;

int main(void) {
    SCB_EnableICache();
    SCB_EnableDCache();

    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART3_UART_Init();

    // stdout is not a tty under nosys, so newlib would fully buffer it and
    // printf output would only reach the UART on fflush. Flush on every '\n'.
    static char stdout_buf[256];
    setvbuf(stdout, stdout_buf, _IOLBF, sizeof(stdout_buf));

    tinyml_app_main();
    while (1) {}
}

// 216 MHz from the internal 16 MHz HSI. HSI is used instead of the 8 MHz
// ST-LINK MCO (HSE bypass) because the default stm32f7xx_hal_conf.h
// assumes HSE_VALUE = 25 MHz, which would corrupt SystemCoreClock.
void SystemClock_Config(void) {
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    osc.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    osc.HSIState = RCC_HSI_ON;
    osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    osc.PLL.PLLM = 8;   // 16 MHz / 8   = 2 MHz
    osc.PLL.PLLN = 216; // 2 MHz * 216  = 432 MHz
    osc.PLL.PLLP = RCC_PLLP_DIV2; // 432 / 2 = 216 MHz SYSCLK
    osc.PLL.PLLQ = 9;   // 432 / 9 = 48 MHz
#if defined(RCC_PLLCFGR_PLLR)
    osc.PLL.PLLR = 2;
#endif
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) {
        Error_Handler();
    }

    if (HAL_PWREx_EnableOverDrive() != HAL_OK) {
        Error_Handler();
    }

    clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                    RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider = RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider = RCC_HCLK_DIV4; // 54 MHz
    clk.APB2CLKDivider = RCC_HCLK_DIV2; // 108 MHz
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_7) != HAL_OK) {
        Error_Handler();
    }
}

void MX_GPIO_Init(void) {
    __HAL_RCC_GPIOD_CLK_ENABLE();
}

void MX_USART3_UART_Init(void) {
    huart3.Instance = USART3;
    huart3.Init.BaudRate = 115200;
    huart3.Init.WordLength = UART_WORDLENGTH_8B;
    huart3.Init.StopBits = UART_STOPBITS_1;
    huart3.Init.Parity = UART_PARITY_NONE;
    huart3.Init.Mode = UART_MODE_TX_RX;
    huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart3.Init.OverSampling = UART_OVERSAMPLING_16;
    huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart3) != HAL_OK) {
        Error_Handler();
    }
}

// Called by HAL_UART_Init
void HAL_UART_MspInit(UART_HandleTypeDef *huart) {
    if (huart->Instance != USART3) {
        return;
    }

    __HAL_RCC_USART3_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();

    GPIO_InitTypeDef gpio = {0};
    gpio.Pin = GPIO_PIN_8 | GPIO_PIN_9; // PD8 = TX, PD9 = RX
    gpio.Mode = GPIO_MODE_AF_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = GPIO_AF7_USART3;
    HAL_GPIO_Init(GPIOD, &gpio);
}

// HAL_Init enables the SysTick interrupt; without this handler the startup
// file's Default_Handler (infinite loop) would hang the MCU on the first tick.
void SysTick_Handler(void) {
    HAL_IncTick();
}

void Error_Handler(void) {
    __disable_irq();
    while (1) {}
}
