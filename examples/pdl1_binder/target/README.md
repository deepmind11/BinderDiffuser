# PD-L1 target structure

This directory holds the target PDB used by `examples/pdl1_binder/config.yaml`.

`pdl1_5o45.pdb` should be the PD-L1 IgV domain (chain B from PDB 5O45, residues
18-134). It is excluded from the repo by default to keep the clone small;
download it on first use:

```bash
mkdir -p examples/pdl1_binder/target
wget https://files.rcsb.org/download/5O45.pdb \
    -O examples/pdl1_binder/target/5o45_full.pdb

# Trim to chain B (the PD-L1 IgV domain) using a one-liner:
python -c "
from Bio.PDB import PDBParser, PDBIO, Select
class ChainB(Select):
    def accept_chain(self, c): return c.id == 'B'
io = PDBIO(); io.set_structure(PDBParser(QUIET=True).get_structure('s','examples/pdl1_binder/target/5o45_full.pdb'))
io.save('examples/pdl1_binder/target/pdl1_5o45.pdb', ChainB())
"
```

The full PDB and its license terms are available at
https://www.rcsb.org/structure/5O45.
