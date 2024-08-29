r"""
Calculate current to a langmuir probe using BRL theory.

Meaning of variables from Laframboise (1966):
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| Variable      | Meaning                                                     | Equation        | BRL Code Location |
+===============+=============================================================+=================+===================+
| `X`           | Array where `X`:math:`= x = 1 / r` and :math:`r` is the     | (5.1)           | Global            |
|               | distance from the origin in either spherical or cylindrical |                 |                   |
|               | coordinates normalized by the Debye length of the positive  |                 |                   |
|               | species.                                                    |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `XSQ`         | Array where `XSQ`:math:`x^2 = 1 / r^2`.                     |                 | Global            |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `S`           | Array of equally spaced points that maps to `X` as defined  |                 | Global            |
|               | in `plasmapy.diagnostics.brl.net_spacing`.                  |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `DXDS`        | Array of the derivative of `X` with respect to `S` as       |                 | Global            |
|               | defined in `plasmapy.diagnostics.brl.net_spacing`.          |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `ROP`         | Array of the probe radius normalized to the Debye length    |                 | Global            |
|               | (:math:`\lambda_D`).                                        |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `SCOT`        | Array where `SCOT`:math:`= \sqrt{1 - x^2}`.                 |                 | Global            |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `COOK`        | Array of the mixing coefficients, (:math:`M(r)` in text     |                 | Global            |
|               | equation (5.1)).                                            |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `XI`          | Array of the normalized potential,                          |                 | Global            |
|               | :math:`\chi(r) = \frac{Z_+ e \phi(r)}{k T_+}` (I think).    |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `DXIDS`       | Array of the derivative of `XI` with respect to `S`.        |                 | Global            |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `ETA`         | Net charge density normalized to the positive charge        |                 | Global            |
|               | density at infinity                                         |                 |                   |
|               | (`ETA`:math:`= \eta = \rho / \rho_\infty_+`).               |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `ETAPS`       | Positive charge density normalized to the positive charge   |                 | Global            |
|               | density at infinity                                         |                 |                   |
|               | (`ETAPS`:math:`= \eta_+ = \rho_+ / \rho_\infty_+`).         |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `ETANG`       | Negative charge density normalized to the positive charge   |                 | Global            |
|               | density at infinity                                         |                 |                   |
|               | (`ETANG`:math:`= \eta_- = \rho_- / \rho_\infty_+`).         |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `RHO`         | Charge density normalized to the positive charge density at |                 | Global            |
|               | infinity                                                    |                 |                   |
|               | (`RHO`:math:`= \rho / \rho_\infty_+`).                      |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `OMGAG`       | Normalized squared angular momentum of the locus of extrema,|                 | Global            |
|               | `OMGAG`:math:`= \Omega_G = -\frac{1}{2x} \frac{d\chi}{dx}`  | (9.8)           |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `BETAG`       | Normalized energy of the locus of extrema,                  |                 | Global            |
|               | `BETAG`:math:`= \beta_G = \chi - \frac{x}{2} \frac{d\chi}{dx}`| (9.8)           |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `P`           | Float value of `S` at outermost point (i.e. `s[-1]`).       |                 | Program 1         |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `POLD`        | Float value of `S` at outermost point for prior iteration.  |                 | Program 1         |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `KINGS`       | Boolean whether `P` is larger than the previous `P`         |                 | Program 1         |
|               | (`POLD`). This only matters if reusing `ETA` from the       |                 |                   |
|               | previous iteration.                                         |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `M`           | Number of points in the computation net.                    |                 | Global            |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
| `MP`          | Number of points in the computation net from the prior      |                 | Global            |
|               | iteration.                                                  |                 |                   |
+---------------+-------------------------------------------------------------+-----------------+-------------------+
"""
