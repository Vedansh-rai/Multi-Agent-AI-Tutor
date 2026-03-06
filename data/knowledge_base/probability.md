# Probability & Statistics Formulas

## Basic Probability
- P(A) = favorable outcomes / total outcomes
- P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- P(A | B) = P(A ∩ B) / P(B)
- P(A') = 1 - P(A)

## Independent Events
- P(A ∩ B) = P(A) × P(B)
- P(A | B) = P(A)

## Bayes' Theorem
P(A | B) = P(B | A) × P(A) / P(B)
P(A_i | B) = P(B | A_i) × P(A_i) / Σ P(B | A_j) × P(A_j)

## Permutations & Combinations
- nPr = n! / (n-r)!
- nCr = n! / (r! × (n-r)!)
- nCr = nC(n-r)
- nC0 + nC1 + ... + nCn = 2^n

## Distributions
### Binomial Distribution
- P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
- Mean: μ = np
- Variance: σ² = np(1-p)

### Poisson Distribution
- P(X = k) = (e^(-λ) × λ^k) / k!
- Mean = Variance = λ

## Conditional Probability
- Total Probability: P(B) = Σ P(B | A_i) × P(A_i)
- Multiplication Rule: P(A ∩ B ∩ C) = P(A) × P(B|A) × P(C|A∩B)

## Random Variables
- E[X] = Σ x_i × P(x_i) (discrete)
- E[X] = ∫ x × f(x) dx (continuous)
- Var(X) = E[X²] - (E[X])²
- E[aX + b] = a × E[X] + b
- Var(aX + b) = a² × Var(X)
