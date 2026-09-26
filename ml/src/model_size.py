"""Real flash footprint of the Random Forest, measured from the firmware build.

The forest is exported by emlearn as inlined if/else code, not as a data
array, so the firmware can't sizeof() it like the other models; the node
count estimate in random_forest_classifier.h (RANDOM_FOREST_MODEL_BYTES)
is only an approximation. This measures the compiled code instead, from the
files PlatformIO leaves in .pio/build/rf after 'pio run -e rf':

  - firmware.elf says which forest functions survived linking (the unused
    random_forest_predict_proba is removed by --gc-sections);
  - the object file of model_rf.cc has each function in its own
    section (-ffunction-sections), so the size of the code sections of those
    functions -- plus, on the ESP32 (Xtensa), their .literal sections, where
    the float thresholds live -- is the forest's real footprint.

The firmware keeps the forest in its own function (tinyml_forest_predict,
noinline) so it doesn't get mixed into classify(). Only needs pyelftools.
"""

from pathlib import Path

FOREST_NAME = 'forest'


def _elf_functions(path):
    """{name: size} of the function symbols in an ELF file."""
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection

    with open(path, 'rb') as handle:
        elf = ELFFile(handle)
        functions = {}
        for section in elf.iter_sections():
            if isinstance(section, SymbolTableSection):
                for symbol in section.iter_symbols():
                    if symbol['st_info']['type'] == 'STT_FUNC' and symbol.name:
                        functions[symbol.name] = symbol['st_size']
        return functions


def _section_sizes(path):
    """{section name: size} of an object file."""
    from elftools.elf.elffile import ELFFile

    with open(path, 'rb') as handle:
        return {section.name: section['sh_size'] for section in ELFFile(handle).iter_sections()}


def measure_forest_flash_bytes(build_directory):
    """Flash bytes of the compiled forest, or None if it can't be measured
    (no build, no pyelftools, or an unexpected build layout)."""
    build_directory = Path(build_directory)
    elf_path = build_directory / 'firmware.elf'
    # model_rf.cc since the firmware refactoring, tinyml_app_rf.cc before
    # (an old build can leave both objects behind: the current one wins).
    # PlatformIO's own builder (STM32) names the object model_rf.o, ESP-IDF's
    # CMake build model_rf.cc.obj.
    objects = []
    for source in ('model_rf', 'tinyml_app_rf'):
        objects = sorted({*build_directory.glob(f'**/{source}.o'),
                          *build_directory.glob(f'**/{source}.cc.o*')})
        if objects:
            break
    if not elf_path.exists() or len(objects) != 1:
        return None
    try:
        linked = {name: size for name, size in _elf_functions(elf_path).items()
                  if FOREST_NAME in name}
        sections = _section_sizes(objects[0])
    except (ImportError, OSError, ValueError):
        return None
    if not any('tinyml_forest_predict' in name for name in linked):
        return None

    total = 0
    for name, symbol_size in linked.items():
        code = sections.get(f'.text.{name}')
        if code is None:
            # Built without -ffunction-sections: fall back to the symbol
            # size (on ARM it includes the function's literal pool).
            total += symbol_size
            continue
        total += code + sections.get(f'.literal.{name}', 0)
    return total
