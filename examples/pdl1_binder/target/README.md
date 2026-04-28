# PD-L1 target structure

This directory holds the target PDB used by `examples/pdl1_binder/config.yaml`.

`pdl1_5o45.pdb` should be the PD-L1 IgV domain extracted from PDB 5O45 and
**relabelled to chain B**. PDB 5O45 itself stores PD-L1 on chain A; we
rename it to B so the target chain id does not collide with RFdiffusion's
default binder output (which lands on chain A).

The file is excluded from the repo by default to keep the clone small;
download and trim it on first use:

```bash
mkdir -p examples/pdl1_binder/target
wget https://files.rcsb.org/download/5O45.pdb \
    -O examples/pdl1_binder/target/5o45_full.pdb

python <<'PY'
from Bio.PDB import PDBIO, PDBParser, Select
parser = PDBParser(QUIET=True)
structure = parser.get_structure('s', 'examples/pdl1_binder/target/5o45_full.pdb')
model = next(structure.get_models())

# Drop everything except chain A (PD-L1 IgV) and rename it to chain B.
for cid in [c.id for c in model if c.id != 'A']:
    model.detach_child(cid)
chain_a = model['A']
model.detach_child('A')
chain_a.id = 'B'
model.add(chain_a)

class StandardOnly(Select):
    def accept_residue(self, r): return r.id[0] == ' '

io = PDBIO()
io.set_structure(structure)
io.save('examples/pdl1_binder/target/pdl1_5o45.pdb', StandardOnly())
print('wrote pdl1_5o45.pdb')
PY
```

After trimming, `pdl1_5o45.pdb` contains exactly one chain (B) with
residues 17-145 — the full PD-L1 IgV domain. The default config uses
motif residues 56-65 on chain B (a contiguous loop on the PD-1-binding
face).

The full PDB and its license terms are available at
https://www.rcsb.org/structure/5O45.
