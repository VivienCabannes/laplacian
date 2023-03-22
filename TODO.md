
Implementation:
- Implement the method with a finite number of features, notably for polynomials (could also be useful for random features).
- Implement a neural network approach to estimate the eigenfunctions of the Laplacian operator.

Real world experiments:
- Unrolling the swiss role
- Implement a clustering algorithm 
- Solve MNIST with it
- Sampling: Samples new digit with it?
- Find a set of problem that are solved with diffusion maps, and solve them with klap.

Ablation study:
- Compare the regularization on L or on R.

Testing suite:
- Add test to check for derivatives computations (q and  q').
- Eventually test for graph Laplacian computation with basic example in 1d, 2d, and d dimensions [L = Id, L_reg = 1, R = diag(lambda_i), ...]

