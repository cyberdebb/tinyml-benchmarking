extern void tinyml_app_main(void);

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART3_UART_Init();
    tinyml_app_main();
    while (1) {}
}