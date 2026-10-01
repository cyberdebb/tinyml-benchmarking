# Enables the STM32F767's hardware FPU (double-precision FPv5, D16).
#
# PlatformIO's stm32cube builder only passes -mcpu=cortex-m7, so GCC falls
# back to soft-float and every float operation (SOS filter, features, SVM
# expf, RF comparisons) runs as a library call. The flags can't just go in
# build_flags: PlatformIO sends those to the compiler only, and the linker
# needs them too to pick the hard-float libc/libm, or the link fails with
# "uses VFP register arguments".
#
# With these flags, core_cm7.h sets __FPU_USED and the framework's SystemInit
# turns the FPU on (CPACR) before main().
Import("env")

fpu_flags = ["-mfloat-abi=hard", "-mfpu=fpv5-d16"]

env.Append(
    ASFLAGS=fpu_flags,
    CCFLAGS=fpu_flags,
    LINKFLAGS=fpu_flags,
)
