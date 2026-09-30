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

On ARM (STM32) a function's symbol size already includes its literal pool,
so if the object file can't be found the ELF alone is enough there.

The firmware keeps the forest in its own function (tinyml_forest_predict,
noinline) so it doesn't get mixed into classify(). Only needs pyelftools.
"""

from pathlib import Path

FOREST_NAME = 'forest'


def _elf_functions(path):
    """({name: size} of the function symbols in an ELF file, its machine)."""
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection

    with open(path, 'rb') as handle:
        elf = ELFFile(handle)
        machine = elf['e_machine']
        functions = {}
        for section in elf.iter_sections():
            if isinstance(section, SymbolTableSection):
                for symbol in section.iter_symbols():
                    if symbol['st_info']['type'] == 'STT_FUNC' and symbol.name:
                        functions[symbol.name] = symbol['st_size']
        return functions, machine


def _section_sizes(path):
    """{section name: size} of an object file."""
    from elftools.elf.elffile import ELFFile

    with open(path, 'rb') as handle:
        return {section.name: section['sh_size'] for section in ELFFile(handle).iter_sections()}


def measure_forest_flash_bytes(build_directory):
    """(flash bytes of the compiled forest, None), or (None, reason) if it
    can't be measured (no build, no pyelftools, or an unexpected layout)."""
    build_directory = Path(build_directory)
    elf_path = build_directory / 'firmware.elf'
    # PlatformIO names it model_rf.cc.o, ESP-IDF (CMake) model_rf.cc.obj.
    objects = sorted(path for path in build_directory.rglob('model_rf*')
                     if path.suffix in ('.o', '.obj'))
    if not elf_path.exists():
        return None, f'{elf_path} not found (run "pio run -e rf" in the firmware folder)'
    try:
        functions, machine = _elf_functions(elf_path)
        sections = _section_sizes(objects[0]) if len(objects) == 1 else {}
    except ImportError:
        return None, 'pyelftools is not installed for this Python (python -m pip install pyelftools)'
    except (OSError, ValueError) as error:
        return None, f'could not read the build files ({error})'
    linked = {name: size for name, size in functions.items() if FOREST_NAME in name}
    if not any('tinyml_forest_predict' in name for name in linked):
        return None, 'tinyml_forest_predict is not in firmware.elf (is it the rf build?)'
    if len(objects) != 1 and machine != 'EM_ARM':
        # Without the object file the .literal sections (the thresholds on
        # Xtensa) can't be counted.
        found = ', '.join(str(path) for path in objects) or 'none'
        return None, f'expected one model_rf object file under {build_directory} (found: {found})'

    total = 0
    for name, symbol_size in linked.items():
        code = sections.get(f'.text.{name}')
        if code is None:
            # Built without -ffunction-sections: fall back to the symbol
            # size (on ARM it includes the function's literal pool).
            total += symbol_size
            continue
        total += code + sections.get(f'.literal.{name}', 0)
    return total, None
