
## CSA

chromatic attention
- different attention score for different channels

Adding substructures: 
- $e_{ij}^{S} = 1(i{\sim}j)$, "1" when i&j belongs to the same substructure


## GRIT
Graph inductive bias

learned PE with RWPE as initialization

a new self-attention mechanism
- conditioning on (learned) relative representations
of node-pairs
- update node-pair representations at the same time

Injecting degree information on the ouput of MSA; replace LN with BN(necessary for leveraging degree information)



## MGT

specially designed for macromolecules

wavelet PE

hierarchical model structure
1. Atom Encoder(GPS+wavelet PE)
2. MPNN -> automaticly aggregate cluster(substructure)
3. Vanilla Transformer(among clusters)


## GPTrans

