# # Replicating "What’s the Dollar of Mexico?" in `HyperdimensionalComputing.jl`

using HyperdimensionalComputing
using LinearAlgebra #hide

# Concept hypervectors

COUNTRY = BipolarHV()
CAPITAL = BipolarHV()
MONEY = BipolarHV()

# Holistic representation of countries

USA = BipolarHV()
MEX = BipolarHV()

WDC = BipolarHV()
MXC = BipolarHV()

DOL = BipolarHV()
PES = BipolarHV()

USTATES = (COUNTRY * USA) + (CAPITAL * WDC) + (MONEY * DOL)
MEXICO = (COUNTRY * MEX) + (CAPITAL * MXC) + (MONEY * PES)

#

USTATES ≈ MEXICO

# If we now pair USTATES with MEXICO, we get a bundle that pairs USA with Mexico, Washington DC
# with Mexico City, and dollar with peso, plus noise:

F_UM = USTATES * MEXICO

# What in Mexico corresponds to United States' dollar:

DOL * F_UM ≈ PES

#

SWE = BipolarHV()
STO = BipolarHV()
KRO = BipolarHV()

SWEDEN = (COUNTRY * SWE) + (CAPITAL * STO) + (MONEY * KRO)

#

F_UM = USTATES * MEXICO
F_SU = SWEDEN * USTATES
F_SM = SWEDEN * MEXICO

F_SU * F_UM ≈ F_SM

#

USTATES * DOL ≈ MEXICO * PES

#

USTATES * DOL * MEXICO ≈ PES

#

USTATES * DOL * MEXICO ≈ KRO

#

USTATES * DOL * MEXICO ≈ DOL
