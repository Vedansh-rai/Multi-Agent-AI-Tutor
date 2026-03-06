# JEE Advanced / Mains — Core Formula Reference

## Quadratic Equations

### Quadratic Formula
For ax² + bx + c = 0:
x = (-b ± √(b²-4ac)) / 2a

Discriminant D = b² - 4ac
- D > 0: two distinct real roots
- D = 0: one repeated real root (x = -b/2a)
- D < 0: two complex conjugate roots

### Vieta's Formulas (Quadratic ax² + bx + c = 0)
Sum of roots: α + β = -b/a
Product of roots: α·β = c/a
Sum of squares: α² + β² = (α+β)² - 2αβ = b²/a² - 2c/a
Difference of roots: |α - β| = √D / |a|

### Nature of Quadratic
- Both roots positive: D ≥ 0, -b/a > 0, c/a > 0
- Both roots negative: D ≥ 0, -b/a < 0, c/a > 0
- Roots of opposite sign: c/a < 0 (irrespective of D)

---

## Polynomial Equations

### Vieta's Formulas (Cubic ax³ + bx² + cx + d = 0, roots α,β,γ)
α + β + γ = -b/a
αβ + βγ + γα = c/a
αβγ = -d/a

### Vieta's Formulas (Quartic ax⁴ + bx³ + cx² + dx + e = 0, roots α,β,γ,δ)
Σα = -b/a
Σαβ = c/a
Σαβγ = -d/a
αβγδ = e/a

### Remainder Theorem
f(a) is the remainder when f(x) is divided by (x - a).

### Factor Theorem
(x - a) is a factor of f(x) iff f(a) = 0.

---

## Inequalities

### AM-GM Inequality
For non-negative reals a₁, a₂, ..., aₙ:
(a₁ + a₂ + ... + aₙ)/n ≥ (a₁·a₂·...·aₙ)^(1/n)
Equality holds iff a₁ = a₂ = ... = aₙ.

### Cauchy-Schwarz Inequality
(a₁b₁ + a₂b₂ + ... + aₙbₙ)² ≤ (a₁² + ... + aₙ²)(b₁² + ... + bₙ²)

### Triangle Inequality
|a + b| ≤ |a| + |b|
||a| - |b|| ≤ |a - b|

### Quadratic Inequality ax²+ bx + c > 0
If a > 0 and D < 0: true for all real x.
If a > 0 and D ≥ 0: true for x < α or x > β (where α ≤ β are roots).
If a < 0 and D > 0: true for α < x < β.

---

## Complex Numbers

### Standard Form
z = a + bi, where i² = -1
Modulus: |z| = √(a² + b²)
Argument: arg(z) = arctan(b/a) [adjusted for quadrant]
Conjugate: z̄ = a - bi
|z|² = z·z̄ = a² + b²

### De Moivre's Theorem
(cos θ + i sin θ)ⁿ = cos(nθ) + i sin(nθ)
zⁿ = rⁿ(cos(nθ) + i sin(nθ))

### nth Roots of Unity
The n roots of z^n = 1 are: e^(2πik/n) for k = 0, 1, ..., n-1
Sum of all nth roots of unity = 0 (for n ≥ 2).
Product of all nth roots of unity = (-1)^(n+1).

### Cube Roots of Unity (ω)
ω = e^(2πi/3) = -1/2 + i√3/2
1 + ω + ω² = 0
ω³ = 1

---

## Sequences and Series

### Arithmetic Progression (AP)
General term: aₙ = a + (n-1)d
Sum of n terms: Sₙ = n/2 · [2a + (n-1)d] = n/2 · (a + l)
where l = last term, a = first term, d = common difference.

### Geometric Progression (GP)
General term: aₙ = ar^(n-1)
Sum of n terms: Sₙ = a(rⁿ - 1)/(r - 1) for r ≠ 1; Sₙ = na for r = 1
Sum of infinite GP (|r| < 1): S∞ = a/(1 - r)

### Harmonic Progression (HP)
Terms 1/a, 1/(a+d), 1/(a+2d), ...
nth term of HP = 1 / [a + (n-1)d]

### AM-GM-HM Relationship
AM ≥ GM ≥ HM (for positive reals)
AM = (a+b)/2, GM = √(ab), HM = 2ab/(a+b)
AM · HM = GM²

### Sum Formulas
Σk from 1 to n = n(n+1)/2
Σk² from 1 to n = n(n+1)(2n+1)/6
Σk³ from 1 to n = [n(n+1)/2]²

---

## Permutations & Combinations

### Fundamental Counting Principle
If event A can occur in m ways and event B in n ways, they can occur together in m·n ways.

### Permutations
nPr = n! / (n-r)! = n·(n-1)·...·(n-r+1)

### Combinations
nCr = n! / (r!(n-r)!) = nPr / r!
nC0 = nCn = 1
nCr = nC(n-r)  [symmetry]
nCr + nC(r-1) = (n+1)Cr  [Pascal's rule]

### Derangements
Number of derangements of n objects: D(n) = n! · Σ(-1)^k/k! for k=0..n
D(n) = n! · (1 - 1/1! + 1/2! - 1/3! + ... + (-1)^n/n!)
D(1)=0, D(2)=1, D(3)=2, D(4)=9, D(5)=44

### Circular Permutations
Arrangements of n distinct objects in a circle = (n-1)!
If clockwise and anticlockwise are same = (n-1)!/2

---

## Binomial Theorem

### Binomial Expansion
(x + y)ⁿ = Σ nCr · x^(n-r) · y^r  for r = 0 to n
= nC0·xⁿ + nC1·x^(n-1)·y + ... + nCn·yⁿ

### General Term (rth term from start)
T(r+1) = nCr · x^(n-r) · y^r

### Middle Term
If n is even: middle term is T(n/2 + 1)
If n is odd: two middle terms T((n+1)/2) and T((n+3)/2)

### Binomial Coefficients Sum
Sum of all binomial coefficients: 2ⁿ
Σ nCr (r even) = Σ nCr (r odd) = 2^(n-1)

### Important Approximations
(1 + x)ⁿ ≈ 1 + nx for |x| << 1

---

## Matrices & Determinants

### Determinant of 2×2 matrix
det([[a,b],[c,d]]) = ad - bc

### Determinant of 3×3 matrix (Sarrus / cofactor expansion)
det([[a,b,c],[d,e,f],[g,h,i]]) = a(ei-fh) - b(di-fg) + c(dh-eg)

### Properties of Determinants
det(AB) = det(A)·det(B)
det(Aᵀ) = det(A)
det(kA) = k^n · det(A) for n×n matrix
det(A⁻¹) = 1/det(A)
If any row or column is all zeros, det = 0.
If two rows or two columns are identical, det = 0.

### Inverse of 2×2 matrix
If A = [[a,b],[c,d]], A⁻¹ = (1/det(A)) · [[d,-b],[-c,a]]

### Cramer's Rule (system Ax = b)
xᵢ = det(Aᵢ) / det(A), where Aᵢ is A with ith column replaced by b.

### Rank & Consistency
System Ax = b is consistent iff rank(A) = rank([A|b]).
If rank(A) = rank([A|b]) = n (unknowns): unique solution.
If rank < n: infinitely many solutions.

### Cayley-Hamilton Theorem
Every square matrix satisfies its own characteristic equation.
A² - tr(A)·A + det(A)·I = 0 for 2×2 matrix.

---

## Probability

### Basic Axioms
0 ≤ P(A) ≤ 1
P(S) = 1 (sample space)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(Aᶜ) = 1 - P(A)

### Conditional Probability
P(A|B) = P(A ∩ B) / P(B), P(B) > 0

### Bayes' Theorem
P(Aᵢ|B) = P(B|Aᵢ)·P(Aᵢ) / Σⱼ P(B|Aⱼ)·P(Aⱼ)

### Independent Events
P(A ∩ B) = P(A)·P(B)

### Binomial Distribution
P(X = k) = nCk · p^k · (1-p)^(n-k)
Mean = np, Variance = np(1-p)

### Hypergeometric Distribution (sampling without replacement)
P(X = k) = C(K,k)·C(N-K, n-k) / C(N,n)
N = population, K = successes in population, n = sample size

### Addition Rule for Mutually Exclusive Events
P(A ∪ B) = P(A) + P(B)

---

## Number System

### Divisibility Rules
Div by 2: last digit even
Div by 3: digit sum divisible by 3
Div by 4: last two digits form a number divisible by 4
Div by 9: digit sum divisible by 9
Div by 11: alternating digit sum divisible by 11

### HCF and LCM
HCF(a,b) × LCM(a,b) = a × b
HCF(a,b,c) × LCM(a,b,c) = a × b × c only when all three are pairwise coprime.

### Logarithm Laws
log_b(xy) = log_b(x) + log_b(y)
log_b(x/y) = log_b(x) - log_b(y)
log_b(xᵃ) = a·log_b(x)
log_b(x) = log_a(x) / log_a(b)  [change of base]
log_b(b) = 1, log_b(1) = 0

### Floor and Ceiling
⌊x⌋ = largest integer ≤ x
⌈x⌉ = smallest integer ≥ x
⌊x⌋ + ⌊x + 1/2⌋ = ⌊2x⌋  [Hermite's identity]
