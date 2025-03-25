# Things that come to mind:

### Signed distance fields on manifolds
 - Interesting for spatial queries / raymarching / sphere-tracing / constrained optimization (barrier potential)

### Bounding volume hierarchies / spatial partitioning on manifolds
Are usually done with axis-aligned bounding-boxes (AABB). Perhaps use AABB in coordinate space?
 - random-forest / decision tree on manifold
 - fast nearest-neighbors queries
 - collision detection / volume overlap $\implies$ physics simulations 😍

### Convolution on manifolds
What does it mean? Can we define the n-dim manifold gaussian PDF as the (maybe unique?) object that is a "scaled" version of itself when convolved with itself, just like in flat space? How do we "blur" / smooth a function on a manifold?

### "Compile a metric" to support faster (approximate?) queries
Maybe can do some work up-front to be able to compute faster geodesics

### RRT on manifolds
(**R**apidly exploring **R**andom **T**rees), a method to solve search/planning problems, particularily in robotics motion and path planning.

# Would like to understand / have better intuition of:

### Manifold & geometry:
Uncertainty in geometry, Parallel transport, "Connections", Christoffel symbols, 

### ODE world:
Differentiable ODE solving, Multiple-shooting for BVP, 