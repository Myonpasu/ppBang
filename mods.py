from enum import IntFlag
from itertools import chain
from itertools import combinations


class Mod(IntFlag):
    NF = 1
    EZ = 2
    HD = 8
    HR = 16
    SD = 32
    DT = 64
    HT = 256
    NC = 512  # Only set along with DoubleTime; Nightcore only gives 576.
    FL = 1024
    SO = 4096
    PF = 16384  # Only set along with SuddenDeath; Perfect only gives 16416.


def allowed_mods():
    """Return the set of possible enabled mods, including no-mod."""
    mods = [Mod.EZ, Mod.HD, Mod.HR, Mod.DT, Mod.HT, Mod.FL]
    mod_powerset = chain.from_iterable(combinations(mods, r) for r in range(len(mods) + 1))
    combos = []
    for p in mod_powerset:
        combined_mod = Mod(0)
        for m in p:
            combined_mod |= m
        combos.append(combined_mod)
    allowed = tuple(c for c in combos if valid_mod(c))
    return allowed


def readable_mod(mod):
    flag_mod = Mod(mod)
    if Mod.NC in flag_mod:
        flag_mod &= ~ Mod.DT
    if Mod.PF in flag_mod:
        flag_mod &= ~ Mod.SD
    name = ','.join(m.name for m in Mod if m in flag_mod)
    if not name:
        name = 'NM'
    return name


def valid_mod(mod):
    if (Mod.EZ in mod and Mod.HR in mod) or (Mod.DT in mod and Mod.HT in mod):
        return False
    else:
        return True
