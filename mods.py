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
    NC = 512  # Only set along with DT; NC only gives 576.
    FL = 1024
    SO = 4096
    PF = 16384  # Only set along with SD; PF only gives 16416.


def allowed_mods():
    """Return all possible mods modulo equivalence, including no-mod."""
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


def equivalent_mods(mod):
    mod = Mod(mod)
    base_mods = [mod]
    if Mod.DT in mod:  # DT equivalent to NC.
        base_mods.append(mod ^ Mod.NC)
    sd_mods = [Mod.SD | m for m in base_mods]
    pf_mods = [Mod.PF | m for m in sd_mods]
    equiv_mods = tuple(base_mods + sd_mods + pf_mods)
    return equiv_mods


def readable_mod(mod):
    mod = Mod(mod)
    if Mod.NC in mod:
        mod &= ~ Mod.DT
    if Mod.PF in mod:
        mod &= ~ Mod.SD
    name = ','.join(m.name for m in Mod if m in mod)
    if not name:
        name = 'NM'
    return name


def valid_mod(mod):
    mod = Mod(mod)
    if (Mod.EZ in mod and Mod.HR in mod) or (Mod.DT in mod and Mod.HT in mod):
        return False
    else:
        return True
