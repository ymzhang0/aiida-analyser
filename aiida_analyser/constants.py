import math

H_PLANCK_SI      = 6.62607015E-34
HARTREE_SI       = 4.3597447222071E-18
ELECTRONVOLT_SI  = 1.602176634E-19
tpi              = 2.0 * math.pi
fpi              = 4.0 * math.pi
AUTOEV           = HARTREE_SI / ELECTRONVOLT_SI
AU_SEC           = H_PLANCK_SI/tpi/HARTREE_SI
AU_PS            = AU_SEC * 1.0E+12
AU_TERAHERTZ     = AU_PS
MEV_TO_THZ       = 1.0E-3 / AU_TERAHERTZ / AUTOEV /tpi
THZ_TO_MEV       = 1.0 / MEV_TO_THZ
THZ_TO_CM        = 33.356