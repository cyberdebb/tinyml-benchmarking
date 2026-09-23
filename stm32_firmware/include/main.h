#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f7xx_hal.h"

// NUCLEO-F767ZI: USART3 is wired to the ST-LINK virtual COM port (PD8/PD9)
extern UART_HandleTypeDef huart3;

void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_USART3_UART_Init(void);
void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif // MAIN_H
