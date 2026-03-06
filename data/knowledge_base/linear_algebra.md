# Linear Algebra Formulas & Rules

## Matrix Operations
- (A + B)ᵀ = Aᵀ + Bᵀ
- (AB)ᵀ = BᵀAᵀ
- (A⁻¹)ᵀ = (Aᵀ)⁻¹
- (AB)⁻¹ = B⁻¹A⁻¹

## Determinants
### 2×2 Matrix
det([[a,b],[c,d]]) = ad - bc

### 3×3 Matrix (Sarrus / Cofactor expansion)
det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)

### Properties
- det(AB) = det(A) × det(B)
- det(Aᵀ) = det(A)
- det(kA) = k^n × det(A) for n×n matrix
- det(A⁻¹) = 1/det(A)
- If any row/column is zero → det = 0
- Swapping two rows changes sign of det

## Matrix Inverse
A⁻¹ = adj(A) / det(A)
A × A⁻¹ = I

For 2×2: [[a,b],[c,d]]⁻¹ = (1/(ad-bc)) × [[d,-b],[-c,a]]

## Cramer's Rule
For AX = B:
x_i = det(A_i) / det(A)
where A_i has column i replaced by B

## Eigenvalues & Eigenvectors
- det(A - λI) = 0 → characteristic equation
- (A - λI)v = 0 → eigenvector
- Sum of eigenvalues = trace(A)
- Product of eigenvalues = det(A)

## Rank
- rank(A) = number of non-zero rows in row echelon form
- rank(A) + nullity(A) = n (number of columns)
- System AX = B is consistent iff rank(A) = rank([A|B])

## Systems of Linear Equations
- Unique solution: rank(A) = rank([A|B]) = n
- Infinite solutions: rank(A) = rank([A|B]) < n
- No solution: rank(A) ≠ rank([A|B])
