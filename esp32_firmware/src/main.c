// extern void tinyml_app_main(void);

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"

void app_main(void) {
    // tinyml_app_main();
    printf("Inicializando ESP32-S3...\n");

    // Verifica a quantidade de memória PSRAM disponível
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Memoria PSRAM livre: %d bytes\n", psram_free);

    if (psram_free == 0) {
        printf("ALERTA: PSRAM nao foi alocada!\n");
    }

    // Loop infinito do FreeRTOS
    while(1) {
        printf("Sistema rodando...\n");
        vTaskDelay(2000 / portTICK_PERIOD_MS); // Pausa de 2 segundos
    }
}