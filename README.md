# Lane Emden Solver

This script will solve the Lane-Emden equation, a second-order differential equation. The Lane–Emden equation is a dimensionless form of Poisson's equation for the gravitational potential of a Newtonian self-gravitating, spherically symmetric, polytropic fluid.
The equation is
$$\frac{1}{\xi} \frac{\text{d}}{\text{d}\xi} \left( \xi^2 \frac{\text{d}\theta}{\text{d}\xi^2}\right) = - \theta^n,$$
where n is the polytropic index.

The first step taken to solve it numerically is to split the equation into two coupled, first-order differential equations.

We can rewrite the Lane-Emden equation into the following 

$$ \frac{2}{\xi} \frac{\text{d}\theta}{\text{d}\xi} + \frac{\text{d}^2\theta}{\text{d}\xi^2} = -\theta^n $$

and split it in the following way

$$ 1) \quad \frac{\text{d}\theta}{\text{d}\xi} = - \frac{\phi}{\xi^2} $$

$$ 2) \quad \frac{\text{d}\phi}{\text{d}\xi} = \theta^n \xi^2 $$
