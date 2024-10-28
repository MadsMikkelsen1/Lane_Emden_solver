# Lane Emden Solver

This script will solve the Lane-Emden equation, a second-order differential equation. The Lane–Emden equation is a dimensionless form of Poisson's equation for the gravitational potential of a Newtonian self-gravitating, spherically symmetric, polytropic fluid.
The equation is
$$\begin{align}
\frac{1}{\xi} \frac{\text{d}}{\text{d}\xi} \left( \xi^2 \frac{\text{d}\theta}{\text{d}\xi^2}\right) = - \theta^n,
\end{align}$$
where n is the polytropic index.

The first step taken to solve it numerically is to split the equation into two coupled, first-order differential equations.