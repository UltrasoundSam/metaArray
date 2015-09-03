#       constants.py
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

# This file contain a number of mathematical and physical constants definition
#

from numpy import log
from numpy import sqrt
from numpy import pi

# Ratio between FWHM and sigma of a Gauss curve = sqrt{8*ln2) 
# 2.3548200450309493
GaussFWHM = sqrt(8*log(2))

# speed of light in vacuum (m s-1)
c = 299792458

# Newtonian constant of gravitation (m3 kg-1 s-2)
G = 6.67384e-11

# Planck constant (J s)
h = 6.62606957e-34

# Magnetic constant (N A-2)
mu0 = 4e-7 * pi

# Elementary charge (C) 
e = 1.602176565e-19

# Electron mass (kg)
me = 9.10938291e-31

# Proton mass (kg)
mp = 1.672621777e-27

# Avogadro constant (mol-1)
NA = 6.02214129e23

# Molar gas constant (J mol-1 K-1)
R = 8.3144621

# Boltzmann constant (J K-1)
k = 1.3806488e-23

# Electric constant (F m-1)
epsilon0 = 1 / (mu0 * c ** 2)

# Characteristic impedance of vacuum (ohm)
Z0 = mu0 * c

# h-bar (J s)
hbar = h / (2*pi)
