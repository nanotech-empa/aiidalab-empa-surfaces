import ase
import numpy as np
from ase.build import sort

# Au lattice param used: 4.1691
# dimensions of a rectangular unit cell of Au with z pointing in (111) direction
au_x = 2.94799888  # a/sqrt(2)
au_y = 5.10608384  # a*sqrt(3/2)

# Ag lattice param used: 4.10669535119
# dimensions of a rectangular unit cell of Ag with z pointing in (111) direction
ag_x = 2.90387213  # a/sqrt(2)
ag_y = 5.02965407  # a*sqrt(3/2)

# Cu lattice param used: 3.60103410694
# dimensions of a rectangular unit cell of Cu with z pointing in (111) direction
cu_x = 2.54631564  # a/sqrt(2)
cu_y = 4.41034805  # a*sqrt(3/2)

# And the x,y coordinates of the Au,Ag,Cu atoms in exact symmetric positions of the three layers of the unit cell
au_exact_xz = np.array(
    [
        [[0.00000000, 1.70202795], [1.47399944, 4.25506987]],
        [[0.00000000, 0.00000000], [1.47399944, 2.55304192]],
        [[0.00000000, 3.40405590], [1.47399944, 0.85101397]],
    ]
)

dz_au_bulk = 2.40703101  # Distance between bulk Au layers
dz_au_h = 0.86519004  # Distance between Au and H layer
dz_au_1_2 = 2.45156905  # Distance between Au top layer and 2nd layer
dz_au_2_3 = 2.39444938  # Distance between Au 2nd layer and 3rd layer (bulk)


# And the x,y coordinates of the Au,Ag,Cu atoms in exact symmetric positions of the three layers of the unit cell
ag_exact_xz = au_exact_xz * 4.10669535119 / 4.1691

dz_ag_bulk = dz_au_bulk * 4.10669535119 / 4.1691  # Distance between bulk Ag layers
dz_ag_h = 0.85106910  # Distance between Ag and H layer
dz_ag_1_2 = 2.35156179  # Distance between Ag top layer and 2nd layer
dz_ag_2_3 = 2.37580264  # Distance between Ag 2nd layer and 3rd layer (bulk)


# And the x,y coordinates of the Au,Ag,Cu atoms in exact symmetric positions of the three layers of the unit cell
cu_exact_xz = au_exact_xz * 3.60103410694 / 4.1691

dz_cu_bulk = dz_au_bulk * 3.60103410694 / 4.1691  # Distance between bulk Cu layers
dz_cu_h = 0.86502011  # Distance between Cu and H layer
dz_cu_1_2 = 2.06853511  # Distance between Cu top layer and 2nd layer
dz_cu_2_3 = 2.08397214  # Distance between Cu 2nd layer and 3rd layer (bulk)

h_bn_x = 2.510999063
h_bn_y = 4.349177955
h_bn_z = 6.842897412
h_bn_unit = [
    ["B", 0.5, 1.0 / 6.0, 0.0],
    ["N", 0.0, 1.0 / 3.0, 0.0],
    ["B", 0.0, 2.0 / 3.0, 0.0],
    ["N", 0.5, 5.0 / 6.0, 0.0],
    ["N", 0.5, 1.0 / 6.0, 0.5],
    ["B", 0.0, 1.0 / 3.0, 0.5],
    ["N", 0.0, 2.0 / 3.0, 0.5],
    ["B", 0.5, 5.0 / 6.0, 0.5],
]
PdGa_Lx = 6.94998812643203
PdGa_Ly = 12.0377325469807
PdGa_Lz = 8.511962399582
Pd3_A_unit = [
    ["Pd", 3.474994063216, 2.006288817622, 0.000000000000],
    ["Pd", -0.000000000000, 8.025155270489, 0.000000000000],
    ["Ga", 5.212491094824, 0.257607487340, 0.763806284236],
    ["Ga", 1.091843182881, 1.375912802281, 0.763806284236],
    ["Ga", 4.120647808380, 4.385346118402, 0.763806284236],
    ["Ga", 1.737497031608, 6.276473446922, 0.763806284236],
    ["Ga", 4.566837349660, 7.394779255147, 0.763806284236],
    ["Ga", 0.645653900508, 10.404212391892, 0.763806284236],
    ["Pd", 5.962394976715, 2.576476048253, 1.612733512153],
    ["Pd", 1.737497031608, 3.875347313730, 1.612733512153],
    ["Pd", 6.200084244541, 5.585909184998, 1.612733512153],
    ["Pd", 2.487400706374, 8.595341962991, 1.612733512153],
    ["Pd", 5.212491094824, 9.894213587220, 1.612733512153],
    ["Pd", 2.725090388450, 11.604775099736, 1.612733512153],
    ["Ga", 3.474994063216, 2.006288817622, 2.546779262310],
    ["Ga", -0.000000000000, 8.025155270489, 2.546779262310],
    ["Pd", 3.474994063216, 6.018866273490, 2.837320799861],
    ["Pd", -0.000000000000, 12.037732546981, 2.837320799861],
    ["Ga", 0.645653900508, 2.379057121404, 3.601127084096],
    ["Ga", 5.212491094824, 4.270185167428, 3.601127084096],
    ["Ga", 1.091843182881, 5.388490258149, 3.601127084096],
    ["Ga", 4.120647808380, 8.397923394894, 3.601127084096],
    ["Ga", 1.737497031608, 10.289051440918, 3.601127084096],
    ["Ga", 4.566837349660, 11.407356531639, 3.601127084096],
    ["Pd", 2.487400706374, 0.570187275475, 4.450054312013],
    ["Pd", 5.212491094824, 1.869058675484, 4.450054312013],
    ["Pd", 2.725090388450, 3.579620546752, 4.450054312013],
    ["Pd", 5.962394976715, 6.589053683497, 4.450054312013],
    ["Pd", 1.737497031608, 7.887924590222, 4.450054312013],
    ["Pd", 6.200084244541, 9.598486820242, 4.450054312013],
    ["Ga", 3.474994063216, 6.018866273490, 5.384100062170],
    ["Ga", -0.000000000000, 12.037732546981, 5.384100062170],
    ["Pd", -0.000000000000, 4.012577635244, 5.674641599721],
    ["Pd", 3.474994063216, 10.031443549982, 5.674641599721],
    ["Ga", 4.120647808380, 0.372768460736, 6.438447883957],
    ["Ga", 1.737497031608, 2.263896170430, 6.438447883957],
    ["Ga", 4.566837349660, 3.382201619903, 6.438447883957],
    ["Ga", 0.645653900508, 6.391634397896, 6.438447883957],
    ["Ga", 5.212491094824, 8.282762443920, 6.438447883957],
    ["Ga", 1.091843182881, 9.401067534641, 6.438447883957],
    ["Pd", 6.200084244541, 1.573331729130, 7.287375111874],
    ["Pd", 2.487400706374, 4.582764686499, 7.287375111874],
    ["Pd", 5.212491094824, 5.881635951976, 7.287375111874],
    ["Pd", 2.725090388450, 7.592197823244, 7.287375111874],
    ["Pd", 5.962394976715, 10.601630959989, 7.287375111874],
    ["Pd", 1.737497031608, 11.900502584218, 7.287375111874],
    ["Ga", 0.000000000000, 4.012577635244, 8.221420862031],
    ["Ga", 3.474994063216, 10.031443549982, 8.221420862031],
]
Pd3_A_top = [
    ["Pd", 3.474994063216, 2.006288817622, 8.511962399582],
    ["Pd", 0.000000000000, 8.025155270489, 8.511962399582],
    ["Ga", 5.212491094824, 0.257607487340, 9.275768683818],
    ["Ga", 1.091843182881, 1.375912802281, 9.275768683818],
    ["Ga", 4.120647808380, 4.385346118402, 9.275768683818],
    ["Ga", 1.737497031608, 6.276473446922, 9.275768683818],
    ["Ga", 4.566837349660, 7.394779255147, 9.275768683818],
    ["Ga", 0.645653900508, 10.404212391892, 9.275768683818],
    ["Pd", 5.962394976715, 2.576476048253, 10.124695911735],
    ["Pd", 1.737497031608, 3.875347313730, 10.124695911735],
    ["Pd", 6.200084244541, 5.585909184998, 10.124695911735],
    ["Pd", 2.487400706374, 8.595341962991, 10.124695911735],
    ["Pd", 5.212491094824, 9.894213587220, 10.124695911735],
    ["Pd", 2.725090388450, 11.604775099736, 10.124695911735],
    ["Ga", 3.474994063216, 2.006288817622, 11.058741661892],
    ["Ga", 0.000000000000, 8.025155270489, 11.058741661892],
    ["Pd", 3.474994063216, 6.018866273490, 11.349283199443],
    ["Pd", 0.000000000000, 12.037732546981, 11.349283199443],
    ["Ga", 0.645653900508, 2.379057121404, 12.113090694399],
    ["Ga", 5.212491094824, 4.270185167428, 12.113090694399],
    ["Ga", 1.091843182881, 5.388490258149, 12.113090694399],
    ["Ga", 4.120647808380, 8.397923394894, 12.113090694399],
    ["Ga", 1.737497031608, 10.289051440918, 12.113090694399],
    ["Ga", 4.566837349660, 11.407356531639, 12.113090694399],
    ["Pd", 2.487400706374, 0.570187275475, 12.962016711596],
    ["Pd", 5.212491094824, 1.869058675484, 12.962016711596],
    ["Pd", 2.725090388450, 3.579620546752, 12.962016711596],
    ["Pd", 5.962394976715, 6.589053683497, 12.962016711596],
    ["Pd", 1.737497031608, 7.887924590222, 12.962016711596],
    ["Pd", 6.200084244541, 9.598486820242, 12.962016711596],
    ["Ga", 3.474994063216, 6.018866273490, 13.896061251032],
    ["Ga", -0.000000000000, 12.037732546981, 13.896061251032],
    ["Pd", 0.000000000000, 4.012577635244, 14.186602788582],
    ["Pd", 3.474994063216, 10.031443549982, 14.186602788582],
    ["Ga", 4.120647808380, 0.372768460736, 14.950410283539],
    ["Ga", 1.737497031608, 2.263896170430, 14.950410283539],
    ["Ga", 4.566837349660, 3.382201619903, 14.950410283539],
    ["Ga", 0.645653900508, 6.391634397896, 14.950410283539],
    ["Ga", 5.212491094824, 8.282762443920, 14.950410283539],
    ["Ga", 1.091843182881, 9.401067534641, 14.950410283539],
    ["Pd", 6.200084244541, 1.573331729130, 15.799336300735],
    ["Pd", 2.487400706374, 4.582764686499, 15.799336300735],
    ["Pd", 5.212491094824, 5.881635951976, 15.799336300735],
    ["Pd", 2.725090388450, 7.592197823244, 15.799336300735],
    ["Pd", 5.962394976715, 10.601630959989, 15.799336300735],
    ["Pd", 1.737497031608, 11.900502584218, 15.799336300735],
]

Pd1_A_unit = [
    ["Pd", 1.737497031608, 0.137230153350, 0.000000000000],
    ["Pd", 5.962394976715, 1.436101497303, 0.000000000000],
    ["Pd", 2.725090388450, 4.445534723737, 0.000000000000],
    ["Pd", 5.212491094824, 6.156096236253, 0.000000000000],
    ["Pd", 2.487400706374, 7.454967860482, 0.000000000000],
    ["Pd", 6.200084244541, 10.464400997227, 0.000000000000],
    ["Ga", 1.091843182881, 2.636664653587, 0.848926026455],
    ["Ga", 5.212491094824, 3.754970103061, 0.848926026455],
    ["Ga", 0.645653900508, 5.646097790333, 0.848926026455],
    ["Ga", 4.566837349660, 8.655531285830, 0.848926026455],
    ["Ga", 1.737497031608, 9.773836376551, 0.848926026455],
    ["Ga", 4.120647808380, 11.664964422575, 0.848926026455],
    ["Pd", 3.474994063216, 2.006288817622, 1.612734285136],
    ["Pd", 0.000000000000, 8.025155270489, 1.612734285136],
    ["Ga", 3.474994063216, 6.018866273490, 1.903275110736],
    ["Ga", -0.000000000000, 12.037732546981, 1.903275110736],
    ["Pd", 6.200084244541, 2.439245906114, 2.837321336552],
    ["Pd", 1.737497031608, 4.149807598007, 2.837321336552],
    ["Pd", 5.962394976715, 5.448678863483, 2.837321336552],
    ["Pd", 2.725090388450, 8.458112000229, 2.837321336552],
    ["Pd", 5.212491094824, 10.168674230249, 2.837321336552],
    ["Pd", 2.487400706374, 11.467545136974, 2.837321336552],
    ["Ga", 4.566837349660, 0.630375925653, 3.686247363006],
    ["Ga", 1.737497031608, 1.748681285438, 3.686247363006],
    ["Ga", 4.120647808380, 3.639809152087, 3.686247363006],
    ["Ga", 1.091843182881, 6.649242288832, 3.686247363006],
    ["Ga", 5.212491094824, 7.767547379553, 3.686247363006],
    ["Ga", 0.645653900508, 9.658675425577, 3.686247363006],
    ["Pd", 3.474994063216, 6.018866273490, 4.450053190099],
    ["Pd", -0.000000000000, 12.037732546981, 4.450053190099],
    ["Ga", -0.000000000000, 4.012577635244, 4.740596447288],
    ["Ga", 3.474994063216, 10.031443549982, 4.740596447288],
    ["Pd", 2.725090388450, 0.432957133336, 5.674642673104],
    ["Pd", 5.212491094824, 2.143518959761, 5.674642673104],
    ["Pd", 2.487400706374, 3.442390225237, 5.674642673104],
    ["Pd", 6.200084244541, 6.451823720735, 5.674642673104],
    ["Pd", 1.737497031608, 8.162385233251, 5.674642673104],
    ["Pd", 5.962394976715, 9.461256857480, 5.674642673104],
    ["Ga", 0.645653900508, 1.633520334464, 6.523568699558],
    ["Ga", 4.566837349660, 4.642953291833, 6.523568699558],
    ["Ga", 1.737497031608, 5.761258741307, 6.523568699558],
    ["Ga", 4.120647808380, 7.652386428579, 6.523568699558],
    ["Ga", 1.091843182881, 10.661819565324, 6.523568699558],
    ["Ga", 5.212491094824, 11.780125373549, 6.523568699558],
    ["Pd", -0.000000000000, 4.012577635244, 7.287374526650],
    ["Pd", 3.474994063216, 10.031443549982, 7.287374526650],
    ["Ga", 3.474994063216, 2.006288817622, 7.577917783840],
    ["Ga", 0.000000000000, 8.025155270489, 7.577917783840],
]

Pd1_A_top = [
    ["Pd", 1.737497031608, 0.137230153350, 8.511961578067],
    ["Pd", 5.962394976715, 1.436101497303, 8.511961578067],
    ["Pd", 2.725090388450, 4.445534723737, 8.511961578067],
    ["Pd", 5.212491094824, 6.156096236253, 8.511961578067],
    ["Pd", 2.487400706374, 7.454967860482, 8.511961578067],
    ["Pd", 6.200084244541, 10.464400997227, 8.511961578067],
    ["Ga", 1.091843182881, 2.636664653587, 9.360890036110],
    ["Ga", 5.212491094824, 3.754970103061, 9.360890036110],
    ["Ga", 0.645653900508, 5.646097790333, 9.360890036110],
    ["Ga", 4.566837349660, 8.655531285830, 9.360890036110],
    ["Ga", 1.737497031608, 9.773836376551, 9.360890036110],
    ["Ga", 4.120647808380, 11.664964422575, 9.360890036110],
    ["Pd", 3.474994063216, 2.006288817622, 10.124695863202],
    ["Pd", 0.000000000000, 8.025155270489, 10.124695863202],
    ["Ga", 3.474994063216, 6.018866273490, 10.415236688803],
    ["Ga", -0.000000000000, 12.037732546981, 10.415236688803],
    ["Pd", 6.200084244541, 2.439245906114, 11.349282914619],
    ["Pd", 1.737497031608, 4.149807598007, 11.349282914619],
    ["Pd", 5.962394976715, 5.448678863483, 11.349282914619],
    ["Pd", 2.725090388450, 8.458112000229, 11.349282914619],
    ["Pd", 5.212491094824, 10.168674230249, 11.349282914619],
    ["Pd", 2.487400706374, 11.467545136974, 11.349282914619],
    ["Ga", 4.566837349660, 0.630375925653, 12.198208941073],
    ["Ga", 1.737497031608, 1.748681285438, 12.198208941073],
    ["Ga", 4.120647808380, 3.639809152087, 12.198208941073],
    ["Ga", 1.091843182881, 6.649242288832, 12.198208941073],
    ["Ga", 5.212491094824, 7.767547379553, 12.198208941073],
    ["Ga", 0.645653900508, 9.658675425577, 12.198208941073],
    ["Pd", 3.474994063216, 6.018866273490, 12.962017199754],
    ["Pd", -0.000000000000, 12.037732546981, 12.962017199754],
    ["Ga", -0.000000000000, 4.012577635244, 13.252558025355],
    ["Ga", 3.474994063216, 10.031443549982, 13.252558025355],
    ["Pd", 2.725090388450, 0.432957133336, 14.186604251171],
    ["Pd", 5.212491094824, 2.143518959761, 14.186604251171],
    ["Pd", 2.487400706374, 3.442390225237, 14.186604251171],
    ["Pd", 6.200084244541, 6.451823720735, 14.186604251171],
    ["Pd", 1.737497031608, 8.162385233251, 14.186604251171],
    ["Pd", 5.962394976715, 9.461256857480, 14.186604251171],
    ["Ga", 0.645653900508, 1.633520334464, 15.035530277625],
    ["Ga", 4.566837349660, 4.642953291833, 15.035530277625],
    ["Ga", 1.737497031608, 5.761258741307, 15.035530277625],
    ["Ga", 4.120647808380, 7.652386428579, 15.035530277625],
    ["Ga", 1.091843182881, 10.661819565324, 15.035530277625],
    ["Ga", 5.212491094824, 11.780125373549, 15.035530277625],
    ["Pd", -0.000000000000, 4.012577635244, 15.799336104717],
    ["Pd", 3.474994063216, 10.031443549982, 15.799336104717],
]
PdGa_dict = {
    "PdGa_A_Pd1": {"bulk": Pd1_A_unit, "top": Pd1_A_top},
    "PdGa_A_Pd3": {"bulk": Pd3_A_unit, "top": Pd3_A_top},
}


# Cu(110)O-2x1
Cu_110_O_2x1_Lx = 2.54631563 * 2
Cu_110_O_2x1_Ly = 3.60103411
Cu_110_O_2x1_Lz = 2.54631563
Cu_110_O_2x1_bulk = [
    ["Cu", 0.0, 0.0, 0],
    ["Cu", 2.54631563, 0.0, 0],
    ["Cu", 1.273157815, 1.800517055, 1.273157815],
    ["Cu", 3.819473445, 1.800517055, 1.273157815],
]

Cu_110_O_2x1_top = [
    ["Cu", 0.0, 0.0, 2.54631563],
    ["Cu", 2.54631563, 0.0, 2.54631563],
    ["Cu", 1.273157815, 1.800517055, 3.81947344500],
    ["O", 1.273157815, 0, 3.81947344500],
]

# Au(110)2x1

Au_110_2x1_Lx = 8.3382
Au_110_2x1_Ly = 2.9482596
Au_110_2x1_Lz = 2.9482596
Au_110_2x1_bulk = [
    ["Au", 0.0, 0.0, 0],
    ["Au", 4.1691, 0.0, 0],
    ["Au", 2.08455, 1.474, 1.4741298],
    ["Au", 6.25365, 1.474, 1.4741298],
]

Au_110_2x1_top = [["Au", 0.0, 0.0, 2.9482596]]
# Au_110_2x1_dict={'bulk' : Au_110_2x1_bulk,'top' : Au_110_2x1_top}


# Au(110)3x1

Au_110_3x1_Lx = 8.3382 / 2.0 * 3.0
Au_110_3x1_Ly = 2.9482596
Au_110_3x1_Lz = 2.9482596
Au_110_3x1_bulk = [
    ["Au", 0.0, 0.0, 0],
    ["Au", 4.1691, 0.0, 0],
    ["Au", 8.3382, 0.0, 0],
    ["Au", 2.08455, 1.474, 1.4741298],
    ["Au", 6.25365, 1.474, 1.4741298],
    ["Au", 10.42275, 1.474, 1.4741298],
]

Au_110_3x1_top = [
    ["Au", 0.0, 0.0, 2.9482596],
    ["Au", 4.1691, 0.0, 2.9482596],
    ["Au", 8.3382, 0.0, 2.9482596],
    ["Au", 2.08455, 1.474, 4.4223894],
    ["Au", 10.42275, 1.474, 4.4223894],
    ["Au", 0.0, 0.0, 5.8965192],
]

# Au(110)4x1

Au_110_4x1_Lx = 8.3382 * 2
Au_110_4x1_Ly = 2.9482596
Au_110_4x1_Lz = 2.9482596
Au_110_4x1_bulk = [
    ["Au", 0.0, 0.0, 0],
    ["Au", 4.1691, 0.0, 0],
    ["Au", 8.3382, 0.0, 0],
    ["Au", 12.5073, 0.0, 0],
    ["Au", 2.08455, 1.474, 1.4741298],
    ["Au", 6.25365, 1.474, 1.4741298],
    ["Au", 10.42275, 1.474, 1.4741298],
    ["Au", 14.59185, 1.474, 1.4741298],
]

Au_110_4x1_top = [
    ["Au", 0.0, 0.0, 2.9482596],
    ["Au", 4.1691, 0.0, 2.9482596],
    ["Au", 12.5073, 0.0, 2.9482596],
    ["Au", 2.08455, 1.474, 4.4223894],
    ["Au", 14.59185, 1.474, 4.4223894],
    ["Au", 0.0, 0.0, 5.8965192],
]

# NaCl(100)
NaCl_100_Lx = 5.6142796  # DZVP-MOLOPT-PBE-GTH-q1 UZH
NaCl_100_Lz = 5.6142796
NaCl_100_top = [
    ["Na", 0.0, 0.0, NaCl_100_Lz],
    ["Na", NaCl_100_Lx / 2, NaCl_100_Lx / 2, NaCl_100_Lz],
    ["Cl", NaCl_100_Lx / 2, 0, NaCl_100_Lz + 0.1],
    ["Cl", 0, NaCl_100_Lx / 2, NaCl_100_Lz + 0.1],
]
NaCl_100_unit = [
    ["Na", 0.0, 0.0, 0],
    ["Na", NaCl_100_Lx / 2, NaCl_100_Lx / 2, 0],
    ["Na", 0, NaCl_100_Lx / 2, NaCl_100_Lz / 2],
    ["Na", NaCl_100_Lx / 2, 0, NaCl_100_Lz / 2],
    ["Cl", NaCl_100_Lx / 2, 0, 0],
    ["Cl", 0, NaCl_100_Lx / 2, 0],
    ["Cl", 0, 0, NaCl_100_Lz / 2],
    ["Cl", NaCl_100_Lx / 2, NaCl_100_Lx / 2, NaCl_100_Lz / 2],
]

slab_lx_ly = {
    "Au(111)": [au_x, au_y],
    "Au(110)2x1": [Au_110_2x1_Lx, Au_110_2x1_Ly],
    "Au(110)3x1": [Au_110_3x1_Lx, Au_110_3x1_Ly],
    "Au(110)4x1": [Au_110_4x1_Lx, Au_110_4x1_Ly],
    "Cu(110)O-2x1": [Cu_110_O_2x1_Lx, Cu_110_O_2x1_Ly],
    "Ag(111)": [ag_x, ag_y],
    "Cu(111)": [cu_x, cu_y],
    "NaCl(100)": [NaCl_100_Lx, NaCl_100_Lx],
    "PdGa_A_Pd1": [PdGa_Lx, PdGa_Ly],
    "PdGa_A_Pd3": [PdGa_Lx, PdGa_Ly],
    "hBN": [h_bn_x, h_bn_y],
}


def guess_slab_size(mol, which_surf):
    cx = np.amax(mol.positions[:, 0]) - np.amin(mol.positions[:, 0]) + 12
    cy = np.amax(mol.positions[:, 1]) - np.amin(mol.positions[:, 1]) + 12
    nx = int(round(cx / slab_lx_ly[which_surf][0])) + 1
    ny = int(round(cy / slab_lx_ly[which_surf][1])) + 1
    return nx, ny


def prepare_slab(mol, dx, dy, dz, phi, nx, ny, nz, which_surf):
    # Au(110)2x1 SECTION
    if "Au(110)2x1" in which_surf:
        au = ase.Atoms()
        for a in Au_110_2x1_bulk:
            au.append(ase.Atom(a[0], (float(a[1]), float(a[2]), float(a[3]))))
        au.cell = (Au_110_2x1_Lx, Au_110_2x1_Ly, Au_110_2x1_Lz)
        au.pbc = (True, True, True)
        # Build 1x1xnz bulk
        au = au.repeat((1, 1, nz))

        # Add top layer
        for a in Au_110_2x1_top:
            au.append(
                ase.Atom(
                    a[0],
                    (float(a[1]), float(a[2]), float(a[3]) + (nz - 1) * Au_110_2x1_Lz),
                )
            )

        # Add to cell thickness of bulk part + vacuum
        au.cell = (Au_110_2x1_Lx, Au_110_2x1_Ly, nz * Au_110_2x1_Lz + 2.9482596 + 40.0)

        # replicate by nx and ny
        au = au.repeat((nx, ny, 1))
        the_slab = sort(au, tags=au.get_positions()[:, 2] * -1)
        the_slab.positions += np.array((0, 0, 10))
        slab_z_max = np.max(the_slab.positions[:, 2])

        cx, cy, cz = the_slab.cell.diagonal()

    # Au(110)3x1 SECTION
    elif "Au(110)3x1" in which_surf:
        au = ase.Atoms()
        for a in Au_110_3x1_bulk:
            au.append(ase.Atom(a[0], (float(a[1]), float(a[2]), float(a[3]))))
        au.cell = (Au_110_3x1_Lx, Au_110_3x1_Ly, Au_110_3x1_Lz)
        au.pbc = (True, True, True)
        # build 1x1xnz bulk
        au = au.repeat((1, 1, nz))

        # Add top layer
        for a in Au_110_3x1_top:
            au.append(
                ase.Atom(
                    a[0],
                    (float(a[1]), float(a[2]), float(a[3]) + (nz - 1) * Au_110_3x1_Lz),
                )
            )

        # Add to cell thickness of bulk part + vacuum
        au.cell = (Au_110_3x1_Lx, Au_110_3x1_Ly, nz * Au_110_3x1_Lz + 2.9482596 + 40.0)

        # Replicate by nx and ny
        au = au.repeat((nx, ny, 1))
        the_slab = sort(au, tags=au.get_positions()[:, 2] * -1)
        the_slab.positions += np.array((0, 0, 10))
        slab_z_max = np.max(the_slab.positions[:, 2])

        cx, cy, cz = the_slab.cell.diagonal()

    # Au(110)4x1 SECTION.
    elif "Au(110)4x1" in which_surf:
        au = ase.Atoms()
        for a in Au_110_4x1_bulk:
            au.append(ase.Atom(a[0], (float(a[1]), float(a[2]), float(a[3]))))
        au.cell = (Au_110_4x1_Lx, Au_110_4x1_Ly, Au_110_4x1_Lz)
        au.pbc = (True, True, True)
        # build 1x1xnz bulk
        au = au.repeat((1, 1, nz))

        # Add top layer.
        for a in Au_110_4x1_top:
            au.append(
                ase.Atom(
                    a[0],
                    (float(a[1]), float(a[2]), float(a[3]) + (nz - 1) * Au_110_4x1_Lz),
                )
            )

        # Add to cell thickness of bulk part + vacuum
        au.cell = (Au_110_4x1_Lx, Au_110_4x1_Ly, nz * Au_110_4x1_Lz + 2.9482596 + 40.0)

        # Replicate by nx and ny.
        au = au.repeat((nx, ny, 1))
        the_slab = sort(au, tags=au.get_positions()[:, 2] * -1)
        the_slab.positions += np.array((0, 0, 10))
        slab_z_max = np.max(the_slab.positions[:, 2])

        cx, cy, cz = the_slab.cell.diagonal()

    # Cu(110)O-2x1 SECTION.
    elif "Cu(110)O-2x1" in which_surf:
        cu = ase.Atoms()
        for a in Cu_110_O_2x1_bulk:
            cu.append(ase.Atom(a[0], (float(a[1]), float(a[2]), float(a[3]))))
        cu.cell = (Cu_110_O_2x1_Lx, Cu_110_O_2x1_Ly, Cu_110_O_2x1_Lz)
        cu.pbc = (True, True, True)
        #  Build 1x1xnz bulk.
        cu = cu.repeat((1, 1, nz))

        # Add top layer.
        for a in Cu_110_O_2x1_top:
            cu.append(
                ase.Atom(
                    a[0],
                    (
                        float(a[1]),
                        float(a[2]),
                        float(a[3]) + (nz - 1) * Cu_110_O_2x1_Lz,
                    ),
                )
            )

        # Add to cell thickness of bulk part + vacuum.
        cu.cell = (
            Cu_110_O_2x1_Lx,
            Cu_110_O_2x1_Ly,
            nz * Cu_110_O_2x1_Lz + 2.54631563 + 40.0,
        )

        # Replicate by nx and ny.
        cu = cu.repeat((nx, ny, 1))
        the_slab = sort(cu, tags=cu.get_positions()[:, 2] * -1)
        the_slab.positions += np.array((0, 0, 10))
        slab_z_max = np.max(the_slab.positions[:, 2])

        cx, cy, cz = the_slab.cell.diagonal()

    # PdGa SECTION.
    elif "PdGa_A_Pd" in which_surf:
        pdga = ase.Atoms()
        for a in PdGa_dict[which_surf]["bulk"]:
            pdga.append(ase.Atom(a[0], (float(a[1]), float(a[2]), float(a[3]))))
        pdga.cell = (PdGa_Lx, PdGa_Ly, PdGa_Lz)
        pdga.pbc = (True, True, True)
        # build 1x1xnz bulk
        pdga = pdga.repeat((1, 1, nz))

        # Add top layers with Pd3 termination.
        for a in PdGa_dict[which_surf]["top"]:
            pdga.append(
                ase.Atom(
                    a[0], (float(a[1]), float(a[2]), float(a[3]) + (nz - 1) * PdGa_Lz)
                )
            )

        # Add to cell thickness of Pd3 part + vacuum.
        pdga.cell = (PdGa_Lx, PdGa_Ly, nz * PdGa_Lz + 7.287373901153 + 40.0)

        # Replicate by nx and ny.
        pd3 = pdga.repeat((nx, ny, 1))
        the_slab = sort(pd3, tags=pd3.get_positions()[:, 2] * -1)
        the_slab.positions += np.array((0, 0, 10))
        slab_z_max = np.max(the_slab.positions[:, 2])

        cx, cy, cz = the_slab.cell.diagonal()
    # hBN
    elif which_surf == "hBN":
        hbn = ase.Atoms()
        for a in h_bn_unit:
            hbn.append(
                ase.Atom(
                    a[0],
                    (float(a[1]) * h_bn_x, float(a[2]) * h_bn_y, float(a[3]) * h_bn_z),
                )
            )
        hbn.cell = (h_bn_x, h_bn_y, h_bn_z)
        hbn.pbc = (True, True, True)
        # build nx x ny x nz bulk
        hbn = hbn.repeat((nx, ny, nz))

        # Add to cell vacuum.
        hbn.cell = (h_bn_x * nx, h_bn_y * ny, h_bn_z * nz + 40.0)
        the_slab = sort(hbn, tags=hbn.get_positions()[:, 2] * -1)
        the_slab.positions += np.array((0, 0, 10))
        slab_z_max = np.max(the_slab.positions[:, 2])

        cx, cy, cz = the_slab.cell.diagonal()
    # NaCl(100)
    elif which_surf == "NaCl(100)":
        nacl = ase.Atoms()
        for a in NaCl_100_unit:
            nacl.append(
                ase.Atom(
                    a[0],
                    (
                        float(a[1]),
                        float(a[2]),
                        float(a[3]),
                    ),
                )
            )
        nacl.cell = (NaCl_100_Lx, NaCl_100_Lx, NaCl_100_Lz)
        nacl.pbc = (True, True, True)
        # build a 1 x 1 x nz bulk
        nacl = nacl.repeat((1, 1, nz))
        nacl.cell = (NaCl_100_Lx, NaCl_100_Lx, NaCl_100_Lz * nz + 40.0)
        # Add top layers with NaCl termination.
        for a in NaCl_100_top:
            nacl.append(
                ase.Atom(
                    a[0],
                    (float(a[1]), float(a[2]), float(a[3]) + (nz - 1) * NaCl_100_Lz),
                )
            )
        # replicate nx x ny
        nacl = nacl.repeat((nx, ny, 1))

        # Add to cell vacuum.
        nacl.cell = (NaCl_100_Lx * nx, NaCl_100_Lx * ny, NaCl_100_Lz * nz + 40.0)
        the_slab = sort(nacl, tags=nacl.get_positions()[:, 2] * -1)
        the_slab.positions += np.array((0, 0, 10))
        slab_z_max = np.max(the_slab.positions[:, 2])

        cx, cy, cz = the_slab.cell.diagonal()
    # NOBLE metals (111).
    else:
        if which_surf == "Au(111)":
            lx = au_x
            ly = au_y
            dz_bulk = dz_au_bulk
            dz_h = dz_au_h
            dz_2_3 = dz_au_2_3
            dz_1_2 = dz_au_1_2
            exact_xz = au_exact_xz
            ida = 79
        elif which_surf == "Ag(111)":
            lx = ag_x
            ly = ag_y
            dz_bulk = dz_ag_bulk
            dz_h = dz_ag_h
            dz_2_3 = dz_ag_2_3
            dz_1_2 = dz_ag_1_2
            exact_xz = ag_exact_xz
            ida = 47
        elif which_surf == "Cu(111)":
            lx = cu_x
            ly = cu_y
            dz_bulk = dz_cu_bulk
            dz_h = dz_cu_h
            dz_2_3 = dz_cu_2_3
            dz_1_2 = dz_cu_1_2
            exact_xz = cu_exact_xz
            ida = 29

        # Build up the unit cell of the gold slab with nz Au layers
        the_slab = []
        layer_z = 10.0
        # H layer in au lattice x-y positions
        the_slab.append([1, exact_xz[0, 0, 0], exact_xz[0, 0, 1], layer_z])
        the_slab.append([1, exact_xz[0, 1, 0], exact_xz[0, 1, 1], layer_z])
        # nz bulk layers
        for i in range(nz):
            if i == 0:
                layer_z += dz_h
            elif i == nz - 2:
                layer_z += dz_2_3
            elif i == nz - 1:
                layer_z += dz_1_2
            else:
                layer_z += dz_bulk
            xy_pos = exact_xz[(i + 1) % 3]
            the_slab.append([ida, xy_pos[0, 0], xy_pos[0, 1], layer_z])
            the_slab.append([ida, xy_pos[1, 0], xy_pos[1, 1], layer_z])

        the_slab = np.array(the_slab)

        # determine cell size
        vac_size = 40.0
        slab_z_max = np.max(the_slab[:, 3])
        slab_z_extent = slab_z_max - np.min(the_slab[:, 3])

        cz = slab_z_extent + vac_size
        cx = nx * lx
        cy = ny * ly

        # generate gold slab based on the rectangular unit cell
        the_slab_raw = []
        for i in range(nx):
            for j in range(ny):
                shift = np.array([0, i * lx, j * ly, 0])
                the_slab_raw.append(the_slab + shift)

        the_slab_raw = np.concatenate(the_slab_raw)
        the_slab = ase.Atoms(numbers=the_slab_raw[:, 0], positions=the_slab_raw[:, 1:])
        the_slab = sort(the_slab, tags=the_slab.get_positions()[:, 2] * -1)

    # print("Cell ABC: %f, %f, %f"%(cx, cy, cz))
    # print("#atoms: %d"%(len(the_slab)))

    # rotate molecule
    mol.rotate(phi, "z")

    mol.cell = (cx, cy, cz)
    mol.pbc = (True, True, True)

    # position molecule a bit above gold slab
    min_mol_slab_dist = 2.3
    avg_mol_slab_dist = 3.2

    mol.center()
    mol_z_min = np.min(mol.positions[:, 2])
    mol_z_avg = np.mean(mol.positions[:, 2])

    min_shift = slab_z_max - mol_z_min + min_mol_slab_dist
    avg_shift = slab_z_max - mol_z_avg + avg_mol_slab_dist

    mol.positions[:, 2] += np.max([min_shift, avg_shift])

    # translate molecule
    mol.positions += np.array([dx, dy, dz])

    return the_slab
