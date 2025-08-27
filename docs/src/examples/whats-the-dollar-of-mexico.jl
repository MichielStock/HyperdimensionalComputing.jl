# # Replicating "What’s the Dollar of Mexico?" in `HyperdimensionalComputing.jl`

using HyperdimensionalComputing
using LinearAlgebra #hide

HyperdimensionalComputing.similarity(u::T, v::T) where {T <: BipolarHDV} = dot(u, v) / (norm(u) * norm(v)) #hide
HyperdimensionalComputing.isapprox(u::T, v::T) where {T <: BipolarHDV} = similarity(u, v) > 3 / sqrt(length(v)) #hide
HyperdimensionalComputing.hash(u::AbstractHDV) = hash(u.v, hash(typeof(v)))
HyperdimensionalComputing.isequal(u::T, v::T) where {T <: AbstractHDV} = hash(u) == hash(v)

# Concept hypervectors

COUNTRY = BipolarHDV()
CAPITAL = BipolarHDV()
MONEY = BipolarHDV()

# Holistic representation of countries

USA = BipolarHDV()
MEX = BipolarHDV()

WDC = BipolarHDV()
MXC = BipolarHDV()

DOL = BipolarHDV()
PES = BipolarHDV()

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

SWE = BipolarHDV()
STO = BipolarHDV()
KRO = BipolarHDV()

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
