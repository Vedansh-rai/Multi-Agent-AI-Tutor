# Algebra Identities & Formulas

## Quadratic Formula
For ax² + bx + c = 0:
x = (-b ± √(b² - 4ac)) / (2a)

Discriminant D = b² - 4ac:
- D > 0: Two distinct real roots
- D = 0: One repeated real root
- D < 0: Two complex conjugate roots

## Factoring Identities
- a² - b² = (a + b)(a - b)
- a² + 2ab + b² = (a + b)²
- a² - 2ab + b² = (a - b)²
- a³ + b³ = (a + b)(a² - ab + b²)
- a³ - b³ = (a - b)(a² + ab + b²)
- a³ + b³ + c³ - 3abc = (a + b + c)(a² + b² + c² - ab - bc - ca)

## Arithmetic & Geometric Progressions
AP: a, a+d, a+2d, ...
- nth term: a_n = a + (n-1)d
- Sum: S_n = n/2 × (2a + (n-1)d)

GP: a, ar, ar², ...
- nth term: a_n = a × r^(n-1)
- Sum (finite): S_n = a(1 - r^n) / (1 - r), r ≠ 1
- Sum (infinite, |r| < 1): S = a / (1 - r)

## Binomial Theorem
(a + b)^n = Σ C(n,k) × a^(n-k) × b^k, k = 0 to n
C(n,k) = n! / (k! × (n-k)!)

## Logarithm Rules
- log(ab) = log(a) + log(b)
- log(a/b) = log(a) - log(b)
- log(a^n) = n × log(a)
- log_a(b) = log(b) / log(a)
- a^(log_a(x)) = x

## Polynomial Division
Remainder theorem: f(a) is the remainder when f(x) is divided by (x - a)
Factor theorem: (x - a) is a factor of f(x) if and only if f(a) = 0

## Inequalities
- AM ≥ GM: (a + b) / 2 ≥ √(ab) for a, b > 0
- Cauchy-Schwarz: (Σ a_i × b_i)² ≤ (Σ a_i²)(Σ b_i²)

## Range of f(x) = x² Over an Interval — Interval Analysis Method

⚠️ NEVER square both sides of an inequality directly. x < 2 does NOT imply x² < 4
  (counter-example: x = -3 → x² = 9 > 4).

**Correct method — split at sign-change points:**

| Domain of x | Sub-intervals | Range of x² |
|---|---|---|
| x < 2 | (-∞,0) ∪ [0,2) | (0,∞) ∪ [0,4) = [0,∞) |
| x > -1 | (-1,0) ∪ [0,∞) | (0,1) ∪ [0,∞) = [0,∞) |
| x ≥ 2 | [2,∞) | [4,∞) |
| x < -1 | (-∞,-1) | (1,∞) |

**Key facts about x² = f(x):**
- x² is always ≥ 0
- x² is DECREASING on (-∞, 0] and INCREASING on [0, ∞)
- Minimum of x² is 0, achieved only at x = 0
- On [a, ∞) with a ≥ 0: x² ∈ [a², ∞)
- On (-∞, -a) with a > 0: x² ∈ (a², ∞)
- On [0, a): x² ∈ [0, a²)
- On (-a, 0) with a > 0: x² ∈ (0, a²)

