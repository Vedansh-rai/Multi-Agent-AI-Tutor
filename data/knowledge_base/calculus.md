# Calculus Formulas & Rules

## Limits
- lim(x→0) sin(x)/x = 1
- lim(x→0) (1 - cos(x))/x = 0
- lim(x→0) (e^x - 1)/x = 1
- lim(x→0) ln(1+x)/x = 1
- lim(x→∞) (1 + 1/x)^x = e

## Derivative Rules
- d/dx [x^n] = n × x^(n-1) (Power Rule)
- d/dx [e^x] = e^x
- d/dx [a^x] = a^x × ln(a)
- d/dx [ln(x)] = 1/x
- d/dx [sin(x)] = cos(x)
- d/dx [cos(x)] = -sin(x)
- d/dx [tan(x)] = sec²(x)
- d/dx [sin⁻¹(x)] = 1/√(1-x²)
- d/dx [cos⁻¹(x)] = -1/√(1-x²)
- d/dx [tan⁻¹(x)] = 1/(1+x²)

## Derivative Techniques
- Product Rule: d/dx [f·g] = f'·g + f·g'
- Quotient Rule: d/dx [f/g] = (f'·g - f·g') / g²
- Chain Rule: d/dx [f(g(x))] = f'(g(x)) × g'(x)

## Integration Rules
- ∫ x^n dx = x^(n+1)/(n+1) + C, n ≠ -1
- ∫ 1/x dx = ln|x| + C
- ∫ e^x dx = e^x + C
- ∫ sin(x) dx = -cos(x) + C
- ∫ cos(x) dx = sin(x) + C
- ∫ sec²(x) dx = tan(x) + C
- ∫ 1/(1+x²) dx = tan⁻¹(x) + C
- ∫ 1/√(1-x²) dx = sin⁻¹(x) + C

## Integration Techniques
- By Parts: ∫ u dv = uv - ∫ v du (ILATE/LIATE priority)
- Substitution: let u = g(x), du = g'(x)dx
- Partial Fractions: decompose rational functions

## Applications
- Area under curve: A = ∫[a,b] f(x) dx
- Volume of revolution: V = π ∫[a,b] [f(x)]² dx (disk method)
- Arc length: L = ∫[a,b] √(1 + [f'(x)]²) dx

## Definite Integrals (JEE Special)
- ∫[0,π] x·f(sin(x)) dx = (π/2) ∫[0,π] f(sin(x)) dx
- ∫[0,2a] f(x) dx = 2∫[0,a] f(x) dx if f(2a-x) = f(x)
- Leibniz Rule: d/dx ∫[a(x),b(x)] f(t)dt = f(b(x))·b'(x) - f(a(x))·a'(x)
