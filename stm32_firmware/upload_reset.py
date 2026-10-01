# Flashes with the STM32 held in reset while the ST-LINK connects.
#
# PlatformIO uploads with OpenOCD's board/st_nucleo_f7.cfg, which connects
# without asserting reset (connect_deassert_srst). The energy firmware
# (energy_app.cc) spends most of its time in WFI, where the bus clock is off
# and that connect fails with "init mode failed (unable to connect to the
# target)". Holding NRST during the connect (connect_assert_srst) works
# whatever the running firmware does; "reset" at the end of the upload still
# starts the new firmware.
#
# A post: script, because the stlink upload flags are built by the platform
# after pre: scripts run, and this platform ignores upload_flags for OpenOCD.
Import("env")

RESET_CONFIG = ["-c", "reset_config srst_only srst_nogate connect_assert_srst"]

flags = list(env.get("UPLOADERFLAGS", []))
program = next((i for i, flag in enumerate(flags) if str(flag).startswith("program ")), None)
if env.get("UPLOADER") == "openocd" and program is not None and RESET_CONFIG[1] not in flags:
    # Right before '-c "program ..."', after the board config it overrides.
    flags[program - 1:program - 1] = RESET_CONFIG
    env.Replace(UPLOADERFLAGS=flags)
