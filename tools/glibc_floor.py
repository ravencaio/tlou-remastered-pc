"""Points a Linux binary's newer libm math symbols at their first glibc version.

usage: glibc_floor.py <max glibc, e.g. 2.41> <ELF file>...

glibc 2.43 and 2.44 gave sqrtf, asinf, atan2f, sinh, ... new symbol versions; a program built
against them asks for GLIBC_2.43/2.44 and cannot start on an older system (SteamOS). glibc keeps
the x86-64 originals (GLIBC_2.2.5), so the requirement on a newer version is rewritten in place,
in .gnu.version_r, to GLIBC_2.2.5: every symbol that used it then binds to the original.
A newer version is only rewritten when all its symbols are libm functions of x86-64's first glibc
(MATH below); anything else stops the package (tools/package_linux.sh).
"""
import struct
import sys

SHT_DYNSYM, SHT_VERNEED, SHT_VERSYM = 11, 0x6FFFFFFE, 0x6FFFFFFF
BASE = b"GLIBC_2.2.5"
# libm functions present in GLIBC_2.2.5 on x86-64 (C89/C99 math).
MATH = set("""
acos acosf acosl asin asinf asinl atan atanf atanl atan2 atan2f atan2l cos cosf cosl sin sinf sinl
tan tanf tanl cosh coshf coshl sinh sinhf sinhl tanh tanhf tanhl acosh acoshf acoshl asinh asinhf
asinhl atanh atanhf atanhl exp expf expl exp2 exp2f exp2l expm1 expm1f expm1l log logf logl log10
log10f log10l log1p log1pf log1pl log2 log2f log2l logb logbf logbl pow powf powl sqrt sqrtf sqrtl
cbrt cbrtf cbrtl hypot hypotf hypotl fmod fmodf fmodl remainder remainderf remainderl erf erff erfl
erfc erfcf erfcl lgamma lgammaf lgammal tgamma tgammaf tgammal ldexp ldexpf ldexpl frexp frexpf
frexpl modf modff modfl scalbn scalbnf scalbnl ilogb ilogbf ilogbl
""".split())


def elf_hash(name):
    h = 0
    for c in name:
        h = ((h << 4) + c) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def version_key(text):
    return tuple(int(part) for part in text.split("."))


def fix(path, floor):
    data = bytearray(open(path, "rb").read())
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return []
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    by_type = {s[1]: s for s in sections}
    if SHT_VERNEED not in by_type or SHT_DYNSYM not in by_type:
        return []
    dynsym, versym, verneed = by_type[SHT_DYNSYM], by_type.get(SHT_VERSYM), by_type[SHT_VERNEED]
    strtab = sections[dynsym[6]]
    string = lambda offset: bytes(data[strtab[4] + offset:data.index(0, strtab[4] + offset)])

    # Symbols by version index.
    users = {}
    if versym:
        for i in range(dynsym[5] // 24):
            index, = struct.unpack_from("<H", data, versym[4] + 2 * i)
            name, = struct.unpack_from("<I", data, dynsym[4] + 24 * i)
            users.setdefault(index & 0x7FFF, []).append(string(name).decode())

    base_name = bytes(data[strtab[4]:strtab[4] + strtab[5]]).find(b"\0" + BASE + b"\0")
    changed, refused = [], []
    offset = verneed[4]
    for _ in range(verneed[7]):          # sh_info: number of entries
        _, count, file_name, aux, next_entry = struct.unpack_from("<HHIII", data, offset)
        library = string(file_name).decode()
        aux_offset = offset + aux
        for _ in range(count):
            _, flags, other, name, next_aux = struct.unpack_from("<IHHII", data, aux_offset)
            version = string(name).decode()
            if version.startswith("GLIBC_") and version[6:][:1].isdigit() and \
                    version_key(version[6:]) > version_key(floor):
                symbols = sorted(s for s in users.get(other, []) if s)
                if library.startswith("libm.so") and symbols and all(s in MATH for s in symbols) and base_name >= 0:
                    struct.pack_into("<IHHI", data, aux_offset, elf_hash(BASE), flags, other, base_name + 1)
                    changed.append(f"{version} -> {BASE.decode()}: {' '.join(symbols)}")
                else:
                    refused.append(f"{library} {version}: {' '.join(symbols) or '(no symbol)'}")
            aux_offset += next_aux
        if not next_entry:
            break
        offset += next_entry
    if refused:
        raise SystemExit(f"{path}: needs a glibc newer than {floor} for more than math functions:\n  "
                         + "\n  ".join(refused))
    if changed:
        open(path, "wb").write(data)
    return changed


def main():
    floor, files = sys.argv[1], sys.argv[2:]
    for path in files:
        for line in fix(path, floor):
            print(f"{path.rsplit('/', 1)[-1]}: {line}")


if __name__ == "__main__":
    main()
