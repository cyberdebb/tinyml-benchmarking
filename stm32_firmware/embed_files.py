# Emulates ESP-IDF's `board_build.embed_files` on ststm32, which ignores that
# option. For each listed file an assembly source is generated exposing the
# same symbols ESP-IDF creates:
#   _binary_<name>_start, _binary_<name>_end, _binary_<name>_size
# where <name> is the file name with non-alphanumeric characters replaced by _.

import os
import re

Import("env")

embed_files = env.BoardConfig().get("build.embed_files", "").split()

if embed_files:
    gen_dir = os.path.join(env.subst("$BUILD_DIR"), "embed_files_src")
    os.makedirs(gen_dir, exist_ok=True)

    for rel_path in embed_files:
        path = os.path.join(env.subst("$PROJECT_DIR"), rel_path)
        if not os.path.isfile(path):
            env.Exit("Error: embedded file not found: %s" % path)

        with open(path, "rb") as f:
            data = f.read()

        name = re.sub(r"[^A-Za-z0-9_]", "_", os.path.basename(rel_path))
        sym = "_binary_" + name
        lines = [
            '    .section .rodata.%s, "a", %%progbits' % sym,
            "    .balign 16",
            "    .global %s_start" % sym,
            "    .type %s_start, %%object" % sym,
            "%s_start:" % sym,
        ]
        for i in range(0, len(data), 16):
            lines.append("    .byte " + ",".join("0x%02x" % b for b in data[i:i + 16]))
        lines += [
            "    .global %s_end" % sym,
            "%s_end:" % sym,
            "    .size %s_start, %s_end - %s_start" % (sym, sym, sym),
            "    .global %s_size" % sym,
            "    .set %s_size, %d" % (sym, len(data)),
            "",
        ]
        content = "\n".join(lines)

        # Only rewrite on change so SCons does not rebuild needlessly
        out_path = os.path.join(gen_dir, name + ".S")
        if not os.path.isfile(out_path) or open(out_path).read() != content:
            with open(out_path, "w") as f:
                f.write(content)

    env.BuildSources(os.path.join("$BUILD_DIR", "embed_files"), gen_dir)
