#ifndef BOARD_H
#define BOARD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f7xx_hal.h"

// NUCLEO-F767ZI: USART3 is wired to the ST-LINK virtual COM port (PD8/PD9).
// printf goes to it (_write in board.c) and the beats are read from it.
extern UART_HandleTypeDef huart3;

// Caches, HAL, 216 MHz clock, USART3 at 115200 baud and line-buffered
// stdout. Must run before anything else.
void board_init(void);

void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif // BOARD_H
