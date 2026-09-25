#include "board.h"

extern void tinyml_app_main(void);

int main(void) {
    board_init();
    tinyml_app_main();
    while (1) {}
}
