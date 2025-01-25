# Progress toward Fusion Energy Breakeven and Gain as Measured against the Lawson Criterion

### Description
The purpose of this codebase is to generate the figures and tables in the paper:

S.E. Wurzel, S.C. Hsu, "Progress toward fusion energy breakeven and gain as
measured against the Lawson criterion" Physics of Plasmas 29, 062103 (2022)
https://doi.org/10.1063/5.0083990

The code is writen in Python 3 and is a Jupyterlab notebook.

This code is not optimized for speed. Running the entire notebook may take
upward of 20 minutes. The main bottleneck is calculating large numbers of
fusion reactivities by integration of cross section over a maxwellian velocity
distribution. These could be cached, however for the sake of simplicity
there is currently no caching functionality.

### Dependencies
- LaTeX
- Python 3.X
- Jupyterlab
- matplotlib
- pandas

### Credits
The cross sections for the reactions,

- T(d,n)4He (D + T --> n + α)
- D(d,p)T (D + D --> p + T)
- D(d,n)3He (D + D --> n + 3He)
- 3He(d,p)4He (D + 3He --> p + α)

are from:
H.S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
https://doi.org/10.1088/0029-5515/32/4/I07

The cross sections for the reaction,

- 11B(p,4He)4He4He (p + 11B --> 3α)

is from
W.M. Nevins and R. Swain 2000 Nucl. Fusion 40 865
https://doi.org/10.1088/0029-5515/40/4/310

### License
MIT License, see LICENSE.

- Sam Wurzel, 11 June 2022
