package functions

import (
	"fmt"
	"math"
)

// BenchmarkFunction opisuje benchmark funkciju za optimizaciju, zajedno sa
// njenim analitičkim gradijentom (gde je dostupan) i metapodacima o
// dozvoljenoj dimenzionalnosti i poznatom globalnom minimumu
type BenchmarkFunction struct {
	Name        string
	Evaluate    func(point []float64) float64
	Gradient    func(point []float64) []float64
	MinDim      int     // minimalan broj dimenzija (0 = bez ograničenja)
	MaxDim      int     // maksimalan broj dimenzija (0 = bez ograničenja)
	GlobalMin   float64 // poznata vrednost globalnog minimuma
	GlobalMinAt string  // opis gde je globalni minimum (npr. [0,0])
}

// ValidateDimension vraća grešku ako n nije kompatibilno sa MinDim/MaxDim ograničenjima funkcije
func (bf BenchmarkFunction) ValidateDimension(n int) error {

	if bf.MinDim > 0 && n < bf.MinDim {
		return fmt.Errorf("function %q requires at least %d dimensions, got %d", bf.Name, bf.MinDim, n)
	}
	if bf.MaxDim > 0 && n > bf.MaxDim {
		return fmt.Errorf("function %q supports at most %d dimensions, got %d", bf.Name, bf.MaxDim, n)
	}

	return nil
}

// NumericalGradient aproksimira gradijent funkcije f u tački koristeći centralne razlike sa korakom h=1e-5
func NumericalGradient(f func([]float64) float64, point []float64) []float64 {

	const h = 1e-5

	gradient := make([]float64, len(point))
	for i := range point {
		plus := append([]float64(nil), point...)
		minus := append([]float64(nil), point...)
		plus[i] += h
		minus[i] -= h
		gradient[i] = (f(plus) - f(minus)) / (2 * h)
	}

	return gradient
}

func sphereEvaluate(point []float64) float64 {

	sum := 0.0
	for _, p := range point {
		sum += p * p
	}

	return sum
}

func sphereGradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		gradient[i] = 2 * p
	}

	return gradient
}

func rosenbrockEvaluate(point []float64) float64 {

	sum := 0.0
	for i := 0; i < len(point)-1; i++ {
		term1 := point[i+1] - point[i]*point[i]
		term2 := 1 - point[i]
		sum += 100*term1*term1 + term2*term2
	}

	return sum
}

func rosenbrockGradient(point []float64) []float64 {

	n := len(point)
	gradient := make([]float64, n)

	for i := 0; i < n-1; i++ {
		gradient[i] += -400*point[i]*(point[i+1]-point[i]*point[i]) - 2*(1-point[i])
	}
	for i := 1; i < n; i++ {
		gradient[i] += 200 * (point[i] - point[i-1]*point[i-1])
	}

	return gradient
}

func rastriginEvaluate(point []float64) float64 {

	n := float64(len(point))
	sum := 10 * n
	for _, p := range point {
		sum += p*p - 10*math.Cos(2*math.Pi*p)
	}

	return sum
}

func rastriginGradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		gradient[i] = 2*p + 20*math.Pi*math.Sin(2*math.Pi*p)
	}

	return gradient
}

func ackleyEvaluate(point []float64) float64 {

	const (
		a = 20.0
		b = 0.2
		c = 2 * math.Pi
	)

	n := float64(len(point))

	sumSq := 0.0
	sumCos := 0.0
	for _, p := range point {
		sumSq += p * p
		sumCos += math.Cos(c * p)
	}

	return -a*math.Exp(-b*math.Sqrt(sumSq/n)) - math.Exp(sumCos/n) + a + math.E
}

func ackleyGradient(point []float64) []float64 {

	const (
		a = 20.0
		b = 0.2
		c = 2 * math.Pi
	)

	n := float64(len(point))

	sumSq := 0.0
	sumCos := 0.0
	for _, p := range point {
		sumSq += p * p
		sumCos += math.Cos(c * p)
	}
	meanSq := sumSq / n
	meanCos := sumCos / n
	sqrtMeanSq := math.Sqrt(meanSq)

	gradient := make([]float64, len(point))
	for i, p := range point {
		term1 := 0.0
		if sqrtMeanSq > 1e-12 {
			term1 = a * b * p * math.Exp(-b*sqrtMeanSq) / (sqrtMeanSq * n)
		}
		term2 := c * math.Sin(c*p) * math.Exp(meanCos) / n
		gradient[i] = term1 + term2
	}

	return gradient
}

func himmelblauEvaluate(point []float64) float64 {

	x, y := point[0], point[1]
	term1 := x*x + y - 11
	term2 := x + y*y - 7

	return term1*term1 + term2*term2
}

func himmelblauGradient(point []float64) []float64 {

	x, y := point[0], point[1]
	term1 := x*x + y - 11
	term2 := x + y*y - 7

	return []float64{
		4*x*term1 + 2*term2,
		2*term1 + 4*y*term2,
	}
}

func bealeEvaluate(point []float64) float64 {

	x, y := point[0], point[1]
	term1 := 1.5 - x + x*y
	term2 := 2.25 - x + x*y*y
	term3 := 2.625 - x + x*y*y*y

	return term1*term1 + term2*term2 + term3*term3
}

func bealeGradient(point []float64) []float64 {

	x, y := point[0], point[1]
	term1 := 1.5 - x + x*y
	term2 := 2.25 - x + x*y*y
	term3 := 2.625 - x + x*y*y*y

	gradX := 2*term1*(-1+y) + 2*term2*(-1+y*y) + 2*term3*(-1+y*y*y)
	gradY := 2*term1*x + 2*term2*2*x*y + 2*term3*3*x*y*y

	return []float64{gradX, gradY}
}

func boothEvaluate(point []float64) float64 {

	x, y := point[0], point[1]
	term1 := x + 2*y - 7
	term2 := 2*x + y - 5

	return term1*term1 + term2*term2
}

func boothGradient(point []float64) []float64 {

	x, y := point[0], point[1]
	term1 := x + 2*y - 7
	term2 := 2*x + y - 5

	return []float64{
		2*term1 + 4*term2,
		4*term1 + 2*term2,
	}
}

func matyasEvaluate(point []float64) float64 {

	x, y := point[0], point[1]

	return 0.26*(x*x+y*y) - 0.48*x*y
}

func matyasGradient(point []float64) []float64 {

	x, y := point[0], point[1]

	return []float64{
		0.52*x - 0.48*y,
		0.52*y - 0.48*x,
	}
}

func leviEvaluate(point []float64) float64 {

	x, y := point[0], point[1]

	sin3px := math.Sin(3 * math.Pi * x)
	sin3py := math.Sin(3 * math.Pi * y)
	sin2py := math.Sin(2 * math.Pi * y)

	term1 := sin3px * sin3px
	term2 := (x - 1) * (x - 1) * (1 + sin3py*sin3py)
	term3 := (y - 1) * (y - 1) * (1 + sin2py*sin2py)

	return term1 + term2 + term3
}

func leviGradient(point []float64) []float64 {
	return NumericalGradient(leviEvaluate, point)
}

func schwefelEvaluate(point []float64) float64 {

	n := float64(len(point))
	sum := 418.9829 * n
	for _, p := range point {
		sum -= p * math.Sin(math.Sqrt(math.Abs(p)))
	}

	return sum
}

func schwefelGradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		if p == 0 {
			gradient[i] = 0
			continue
		}
		sqrtAbs := math.Sqrt(math.Abs(p))
		/*
		 d/dx[-x*sin(sqrt|x|)] = -sin(sqrt|x|) - |x|*cos(sqrt|x|)/(2*sqrt|x|),
		 razlog - d(sqrt|x|)/dx nosi faktor sign(x) koji poništava x u članu iz pravila proizvoda, ostavljajući |x| umesto x
		*/
		gradient[i] = -math.Sin(sqrtAbs) - math.Abs(p)*math.Cos(sqrtAbs)/(2*sqrtAbs)
	}

	return gradient
}

func griewankEvaluate(point []float64) float64 {

	sum := 0.0
	product := 1.0
	for i, p := range point {
		sum += p * p / 4000
		product *= math.Cos(p / math.Sqrt(float64(i+1)))
	}

	return sum - product + 1
}

func griewankGradient(point []float64) []float64 {
	return NumericalGradient(griewankEvaluate, point)
}

func styblinskiTangEvaluate(point []float64) float64 {

	sum := 0.0
	for _, p := range point {
		sum += p*p*p*p - 16*p*p + 5*p
	}

	return sum / 2
}

func styblinskiTangGradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		gradient[i] = (4*p*p*p - 32*p + 5) / 2
	}

	return gradient
}

func easomEvaluate(point []float64) float64 {

	x, y := point[0], point[1]
	dx := x - math.Pi
	dy := y - math.Pi

	return -math.Cos(x) * math.Cos(y) * math.Exp(-(dx*dx + dy*dy))
}

func easomGradient(point []float64) []float64 {
	return NumericalGradient(easomEvaluate, point)
}

func crossInTrayEvaluate(point []float64) float64 {

	x, y := point[0], point[1]
	inner := math.Abs(100 - math.Sqrt(x*x+y*y)/math.Pi)

	return -0.0001 * math.Pow(math.Abs(math.Sin(x)*math.Sin(y)*math.Exp(inner))+1, 0.1)
}

func crossInTrayGradient(point []float64) []float64 {
	return NumericalGradient(crossInTrayEvaluate, point)
}

func eggholderEvaluate(point []float64) float64 {

	x, y := point[0], point[1]

	term1 := -(y + 47) * math.Sin(math.Sqrt(math.Abs(x/2+(y+47))))
	term2 := -x * math.Sin(math.Sqrt(math.Abs(x-(y+47))))

	return term1 + term2
}

func eggholderGradient(point []float64) []float64 {
	return NumericalGradient(eggholderEvaluate, point)
}

func bukinN6Evaluate(point []float64) float64 {

	x, y := point[0], point[1]

	return 100*math.Sqrt(math.Abs(y-0.01*x*x)) + 0.01*math.Abs(x+10)
}

func bukinN6Gradient(point []float64) []float64 {
	return NumericalGradient(bukinN6Evaluate, point)
}

func dropWaveEvaluate(point []float64) float64 {

	x, y := point[0], point[1]
	sumSq := x*x + y*y

	return -(1 + math.Cos(12*math.Sqrt(sumSq))) / (0.5*sumSq + 2)
}

func dropWaveGradient(point []float64) []float64 {
	return NumericalGradient(dropWaveEvaluate, point)
}

func schafferN2Evaluate(point []float64) float64 {

	x, y := point[0], point[1]
	sin := math.Sin(x*x - y*y)
	denom := 1 + 0.001*(x*x+y*y)

	return 0.5 + (sin*sin-0.5)/(denom*denom)
}

func schafferN2Gradient(point []float64) []float64 {
	return NumericalGradient(schafferN2Evaluate, point)
}

func sixHumpCamelEvaluate(point []float64) float64 {

	x, y := point[0], point[1]

	return (4-2.1*x*x+x*x*x*x/3)*x*x + x*y + (-4+4*y*y)*y*y
}

func sixHumpCamelGradient(point []float64) []float64 {
	return NumericalGradient(sixHumpCamelEvaluate, point)
}

func threeHumpCamelEvaluate(point []float64) float64 {

	x, y := point[0], point[1]

	return 2*x*x - 1.05*x*x*x*x + x*x*x*x*x*x/6 + x*y + y*y
}

func threeHumpCamelGradient(point []float64) []float64 {

	x, y := point[0], point[1]

	return []float64{
		4*x - 4.2*x*x*x + x*x*x*x*x + y,
		x + 2*y,
	}
}

func mccormickEvaluate(point []float64) float64 {

	x, y := point[0], point[1]

	return math.Sin(x+y) + (x-y)*(x-y) - 1.5*x + 2.5*y + 1
}

func mccormickGradient(point []float64) []float64 {

	x, y := point[0], point[1]
	cosXY := math.Cos(x + y)

	return []float64{
		cosXY + 2*(x-y) - 1.5,
		cosXY - 2*(x-y) + 2.5,
	}
}

func shubertEvaluate(point []float64) float64 {

	x, y := point[0], point[1]

	sumX := 0.0
	sumY := 0.0
	for i := 1; i <= 5; i++ {
		fi := float64(i)
		sumX += fi * math.Cos((fi+1)*x+fi)
		sumY += fi * math.Cos((fi+1)*y+fi)
	}

	return sumX * sumY
}

func shubertGradient(point []float64) []float64 {
	return NumericalGradient(shubertEvaluate, point)
}

const michalewiczM = 10

func michalewiczEvaluate(point []float64) float64 {

	sum := 0.0
	for i, p := range point {
		fi := float64(i + 1)
		u := fi * p * p / math.Pi
		sum += math.Sin(p) * math.Pow(math.Sin(u), 2*michalewiczM)
	}

	return -sum
}

func michalewiczGradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		fi := float64(i + 1)
		u := fi * p * p / math.Pi
		sinU := math.Sin(u)
		cosU := math.Cos(u)

		term1 := math.Cos(p) * math.Pow(sinU, 2*michalewiczM)
		term2 := math.Sin(p) * 2 * michalewiczM * math.Pow(sinU, 2*michalewiczM-1) * cosU * (2 * fi * p / math.Pi)

		gradient[i] = -(term1 + term2)
	}

	return gradient
}

func zakharovSum(point []float64) float64 {

	s := 0.0
	for i, p := range point {
		s += 0.5 * float64(i+1) * p
	}

	return s
}

func zakharovEvaluate(point []float64) float64 {

	sumSq := 0.0
	for _, p := range point {
		sumSq += p * p
	}

	s := zakharovSum(point)

	return sumSq + s*s + s*s*s*s
}

func zakharovGradient(point []float64) []float64 {

	s := zakharovSum(point)
	gradient := make([]float64, len(point))
	for i, p := range point {
		gradient[i] = 2*p + (2*s+4*s*s*s)*0.5*float64(i+1)
	}

	return gradient
}

func dixonPriceEvaluate(point []float64) float64 {

	sum := (point[0] - 1) * (point[0] - 1)
	for i := 1; i < len(point); i++ {
		term := 2*point[i]*point[i] - point[i-1]
		sum += float64(i) * term * term
	}

	return sum
}

func dixonPriceGradient(point []float64) []float64 {
	return NumericalGradient(dixonPriceEvaluate, point)
}

func powellEvaluate(point []float64) float64 {

	sum := 0.0
	for base := 0; base+3 < len(point); base += 4 {
		a, b, c, d := point[base], point[base+1], point[base+2], point[base+3]

		term1 := a + 10*b
		term2 := c - d
		term3 := b - 2*c
		term4 := a - d

		sum += term1*term1 + 5*term2*term2 + term3*term3*term3*term3 + 10*term4*term4*term4*term4
	}

	return sum
}

func powellGradient(point []float64) []float64 {
	return NumericalGradient(powellEvaluate, point)
}

func rotatedHyperEllipsoidEvaluate(point []float64) float64 {

	sum := 0.0
	running := 0.0
	for _, p := range point {
		running += p * p
		sum += running
	}

	return sum
}

func rotatedHyperEllipsoidGradient(point []float64) []float64 {

	n := len(point)
	gradient := make([]float64, n)
	for i, p := range point {
		gradient[i] = 2 * p * float64(n-i)
	}

	return gradient
}

func sumOfDifferentPowersEvaluate(point []float64) float64 {

	sum := 0.0
	for i, p := range point {
		sum += math.Pow(math.Abs(p), float64(i+2))
	}

	return sum
}

func sumOfDifferentPowersGradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		sign := 0.0
		switch {
		case p > 0:
			sign = 1
		case p < 0:
			sign = -1
		}
		gradient[i] = float64(i+2) * sign * math.Pow(math.Abs(p), float64(i+1))
	}

	return gradient
}

func tridEvaluate(point []float64) float64 {

	sum := 0.0
	for _, p := range point {
		sum += (p - 1) * (p - 1)
	}
	for i := 1; i < len(point); i++ {
		sum -= point[i] * point[i-1]
	}

	return sum
}

func tridGradient(point []float64) []float64 {

	n := len(point)
	gradient := make([]float64, n)
	for i, p := range point {
		gradient[i] = 2 * (p - 1)
		if i > 0 {
			gradient[i] -= point[i-1]
		}
		if i < n-1 {
			gradient[i] -= point[i+1]
		}
	}

	return gradient
}

func colvilleEvaluate(point []float64) float64 {

	x0, x1, x2, x3 := point[0], point[1], point[2], point[3]

	term1 := x1 - x0*x0
	term2 := x3 - x2*x2

	return 100*term1*term1 + (1-x0)*(1-x0) + 90*term2*term2 + (1-x2)*(1-x2) +
		10.1*((x1-1)*(x1-1)+(x3-1)*(x3-1)) + 19.8*(x1-1)*(x3-1)
}

func colvilleGradient(point []float64) []float64 {

	x0, x1, x2, x3 := point[0], point[1], point[2], point[3]

	return []float64{
		-400*x0*(x1-x0*x0) - 2*(1-x0),
		200*(x1-x0*x0) + 20.2*(x1-1) + 19.8*(x3-1),
		-360*x2*(x3-x2*x2) - 2*(1-x2),
		180*(x3-x2*x2) + 20.2*(x3-1) + 19.8*(x1-1),
	}
}

const permBeta = 0.5

func permEvaluate(point []float64) float64 {

	n := len(point)
	total := 0.0
	for k := 1; k <= n; k++ {
		inner := 0.0
		for i := 1; i <= n; i++ {
			inner += (math.Pow(float64(i), float64(k)) + permBeta) * (math.Pow(point[i-1]/float64(i), float64(k)) - 1)
		}
		total += inner * inner
	}

	return total
}

func permGradient(point []float64) []float64 {
	return NumericalGradient(permEvaluate, point)
}

func goldsteinPriceEvaluate(point []float64) float64 {

	x, y := point[0], point[1]

	a := (x + y + 1) * (x + y + 1)
	b := 19 - 14*x + 3*x*x - 14*y + 6*x*y + 3*y*y
	c := (2*x - 3*y) * (2*x - 3*y)
	d := 18 - 32*x + 12*x*x + 48*y - 36*x*y + 27*y*y

	return (1 + a*b) * (30 + c*d)
}

func goldsteinPriceGradient(point []float64) []float64 {
	return NumericalGradient(goldsteinPriceEvaluate, point)
}

func bohachevskyN1Evaluate(point []float64) float64 {

	x, y := point[0], point[1]

	return x*x + 2*y*y - 0.3*math.Cos(3*math.Pi*x) - 0.4*math.Cos(4*math.Pi*y) + 0.7
}

func bohachevskyN1Gradient(point []float64) []float64 {

	x, y := point[0], point[1]

	return []float64{
		2*x + 0.9*math.Pi*math.Sin(3*math.Pi*x),
		4*y + 1.6*math.Pi*math.Sin(4*math.Pi*y),
	}
}

func bohachevskyN2Evaluate(point []float64) float64 {

	x, y := point[0], point[1]

	return x*x + 2*y*y - 0.3*math.Cos(3*math.Pi*x)*math.Cos(4*math.Pi*y) + 0.3
}

func bohachevskyN2Gradient(point []float64) []float64 {
	return NumericalGradient(bohachevskyN2Evaluate, point)
}

func bohachevskyN3Evaluate(point []float64) float64 {

	x, y := point[0], point[1]

	return x*x + 2*y*y - 0.3*math.Cos(3*math.Pi*x+4*math.Pi*y) + 0.3
}

func bohachevskyN3Gradient(point []float64) []float64 {

	x, y := point[0], point[1]
	phase := 3*math.Pi*x + 4*math.Pi*y

	return []float64{
		2*x + 0.9*math.Pi*math.Sin(phase),
		4*y + 1.2*math.Pi*math.Sin(phase),
	}
}

func schafferN4Evaluate(point []float64) float64 {

	x, y := point[0], point[1]

	cosTerm := math.Cos(math.Sin(math.Abs(x*x - y*y)))
	denom := 1 + 0.001*(x*x+y*y)

	return 0.5 + (cosTerm*cosTerm-0.5)/(denom*denom)
}

func schafferN4Gradient(point []float64) []float64 {
	return NumericalGradient(schafferN4Evaluate, point)
}

func zettlEvaluate(point []float64) float64 {

	x, y := point[0], point[1]
	u := x*x + y*y - 2*x

	return u*u + x/4
}

func zettlGradient(point []float64) []float64 {

	x, y := point[0], point[1]
	u := x*x + y*y - 2*x

	return []float64{
		4*(x-1)*u + 0.25,
		4 * y * u,
	}
}

var shekelC = [4][10]float64{
	{4, 1, 8, 6, 3, 2, 5, 8, 6, 7},
	{4, 1, 8, 6, 7, 9, 3, 1, 2, 3},
	{4, 1, 8, 6, 3, 2, 5, 8, 6, 7},
	{4, 1, 8, 6, 7, 9, 3, 1, 2, 3},
}

var shekelBeta = [10]float64{0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5}

func shekelEvaluate(point []float64) float64 {

	sum := 0.0
	for i := 0; i < 10; i++ {
		inner := shekelBeta[i]
		for j := 0; j < 4; j++ {
			diff := point[j] - shekelC[j][i]
			inner += diff * diff
		}
		sum += 1 / inner
	}

	return -sum
}

func shekelGradient(point []float64) []float64 {
	return NumericalGradient(shekelEvaluate, point)
}

var hartmann3Alpha = [4]float64{1.0, 1.2, 3.0, 3.2}

var hartmann3A = [4][3]float64{
	{3.0, 10, 30},
	{0.1, 10, 35},
	{3.0, 10, 30},
	{0.1, 10, 35},
}

var hartmann3P = [4][3]float64{
	{0.3689, 0.1170, 0.2673},
	{0.4699, 0.4387, 0.7470},
	{0.1091, 0.8732, 0.5547},
	{0.0382, 0.5743, 0.8828},
}

func hartmann3DEvaluate(point []float64) float64 {

	sum := 0.0
	for i := 0; i < 4; i++ {
		inner := 0.0
		for j := 0; j < 3; j++ {
			diff := point[j] - hartmann3P[i][j]
			inner += hartmann3A[i][j] * diff * diff
		}
		sum += hartmann3Alpha[i] * math.Exp(-inner)
	}

	return -sum
}

func hartmann3DGradient(point []float64) []float64 {
	return NumericalGradient(hartmann3DEvaluate, point)
}

var hartmann6Alpha = [4]float64{1.0, 1.2, 3.0, 3.2}

var hartmann6A = [4][6]float64{
	{10, 3, 17, 3.5, 1.7, 8},
	{0.05, 10, 17, 0.1, 8, 14},
	{3, 3.5, 1.7, 10, 17, 8},
	{17, 8, 0.05, 10, 0.1, 14},
}

var hartmann6P = [4][6]float64{
	{0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
	{0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
	{0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650},
	{0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381},
}

func hartmann6DEvaluate(point []float64) float64 {

	sum := 0.0
	for i := 0; i < 4; i++ {
		inner := 0.0
		for j := 0; j < 6; j++ {
			diff := point[j] - hartmann6P[i][j]
			inner += hartmann6A[i][j] * diff * diff
		}
		sum += hartmann6Alpha[i] * math.Exp(-inner)
	}

	return -sum
}

func hartmann6DGradient(point []float64) []float64 {
	return NumericalGradient(hartmann6DEvaluate, point)
}

func bentCigarEvaluate(point []float64) float64 {

	sum := point[0] * point[0]
	for i := 1; i < len(point); i++ {
		sum += 1e6 * point[i] * point[i]
	}

	return sum
}

func bentCigarGradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	gradient[0] = 2 * point[0]
	for i := 1; i < len(point); i++ {
		gradient[i] = 2e6 * point[i]
	}

	return gradient
}

const (
	weierstrassA    = 0.5
	weierstrassB    = 3.0
	weierstrassKMax = 20
)

func weierstrassEvaluate(point []float64) float64 {

	n := len(point)

	baseSum := 0.0
	for k := 0; k <= weierstrassKMax; k++ {
		baseSum += math.Pow(weierstrassA, float64(k)) * math.Cos(2*math.Pi*math.Pow(weierstrassB, float64(k))*0.5)
	}

	sum := 0.0
	for _, p := range point {
		for k := 0; k <= weierstrassKMax; k++ {
			sum += math.Pow(weierstrassA, float64(k)) * math.Cos(2*math.Pi*math.Pow(weierstrassB, float64(k))*(p+0.5))
		}
	}

	return sum - float64(n)*baseSum
}

func weierstrassGradient(point []float64) []float64 {
	return NumericalGradient(weierstrassEvaluate, point)
}

func salomonEvaluate(point []float64) float64 {

	sumSq := 0.0
	for _, p := range point {
		sumSq += p * p
	}
	r := math.Sqrt(sumSq)

	return 1 - math.Cos(2*math.Pi*r) + 0.1*r
}

func salomonGradient(point []float64) []float64 {

	sumSq := 0.0
	for _, p := range point {
		sumSq += p * p
	}
	r := math.Sqrt(sumSq)

	gradient := make([]float64, len(point))
	if r > 1e-10 {
		factor := (2*math.Pi*math.Sin(2*math.Pi*r) + 0.1) / r
		for i, p := range point {
			gradient[i] = p * factor
		}
	}

	return gradient
}

func alpineN1Evaluate(point []float64) float64 {

	sum := 0.0
	for _, p := range point {
		sum += math.Abs(p*math.Sin(p) + 0.1*p)
	}

	return sum
}

func alpineN1Gradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		inner := p*math.Sin(p) + 0.1*p
		innerDeriv := math.Sin(p) + p*math.Cos(p) + 0.1

		sign := 0.0
		switch {
		case inner > 0:
			sign = 1
		case inner < 0:
			sign = -1
		}
		gradient[i] = sign * innerDeriv
	}

	return gradient
}

func alpineN2Evaluate(point []float64) float64 {

	product := 1.0
	for _, p := range point {
		product *= math.Sqrt(p) * math.Sin(p)
	}

	return -product
}

func alpineN2Gradient(point []float64) []float64 {
	return NumericalGradient(alpineN2Evaluate, point)
}

func periodicEvaluate(point []float64) float64 {

	sumSin := 0.0
	sumSq := 0.0
	for _, p := range point {
		sumSin += math.Sin(p) * math.Sin(p)
		sumSq += p * p
	}

	return 1 + sumSin - 0.1*math.Exp(-sumSq)
}

func periodicGradient(point []float64) []float64 {

	sumSq := 0.0
	for _, p := range point {
		sumSq += p * p
	}
	expTerm := math.Exp(-sumSq)

	gradient := make([]float64, len(point))
	for i, p := range point {
		gradient[i] = math.Sin(2*p) + 0.2*p*expTerm
	}

	return gradient
}

func qingEvaluate(point []float64) float64 {

	sum := 0.0
	for i, p := range point {
		term := p*p - float64(i+1)
		sum += term * term
	}

	return sum
}

func qingGradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		gradient[i] = 4 * p * (p*p - float64(i+1))
	}

	return gradient
}

func katsuuraEvaluate(point []float64) float64 {

	n := len(point)
	fn := float64(n)
	exponent := 10 / math.Pow(fn, 1.2)

	product := 1.0
	for i, p := range point {
		termSum := 0.0
		for k := 1; k <= 32; k++ {
			pow2k := math.Pow(2, float64(k))
			termSum += math.Abs(pow2k*p-math.Round(pow2k*p)) / pow2k
		}
		product *= math.Pow(1+float64(i+1)*termSum, exponent)
	}

	return (10/(fn*fn))*product - 10/(fn*fn)
}

func katsuuraGradient(point []float64) []float64 {
	return NumericalGradient(katsuuraEvaluate, point)
}

const happycatAlpha = 1.0 / 8.0

func happycatEvaluate(point []float64) float64 {

	n := float64(len(point))

	sumSq := 0.0
	sumX := 0.0
	for _, p := range point {
		sumSq += p * p
		sumX += p
	}

	return math.Pow(math.Abs(sumSq-n), 2*happycatAlpha) + (0.5*sumSq+sumX)/n + 0.5
}

func happycatGradient(point []float64) []float64 {
	return NumericalGradient(happycatEvaluate, point)
}

const hgbatAlpha = 1.0 / 4.0

func hgbatEvaluate(point []float64) float64 {

	n := float64(len(point))

	sumSq := 0.0
	sumX := 0.0
	for _, p := range point {
		sumSq += p * p
		sumX += p
	}

	return math.Pow(math.Abs(sumSq-n), 2*hgbatAlpha) + (0.5*sumSq+sumX)/n + 0.5
}

func hgbatGradient(point []float64) []float64 {
	return NumericalGradient(hgbatEvaluate, point)
}

func schwefel221Evaluate(point []float64) float64 {

	maxAbs := 0.0
	for _, p := range point {
		if math.Abs(p) > maxAbs {
			maxAbs = math.Abs(p)
		}
	}

	return maxAbs
}

func schwefel221Gradient(point []float64) []float64 {
	return NumericalGradient(schwefel221Evaluate, point)
}

func schwefel222Evaluate(point []float64) float64 {

	sum := 0.0
	product := 1.0
	for _, p := range point {
		sum += math.Abs(p)
		product *= math.Abs(p)
	}

	return sum + product
}

func schwefel222Gradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		sign := 0.0
		switch {
		case p > 0:
			sign = 1
		case p < 0:
			sign = -1
		}

		productExcluding := 1.0
		for j, q := range point {
			if j != i {
				productExcluding *= math.Abs(q)
			}
		}

		gradient[i] = sign + sign*productExcluding
	}

	return gradient
}

func differentPowersEvaluate(point []float64) float64 {

	n := len(point)
	sum := 0.0
	for i, p := range point {
		exponent := 2 + 4*float64(i)/float64(n-1)
		sum += math.Pow(math.Abs(p), exponent)
	}

	return sum
}

func differentPowersGradient(point []float64) []float64 {

	n := len(point)
	gradient := make([]float64, n)
	for i, p := range point {
		exponent := 2 + 4*float64(i)/float64(n-1)

		sign := 0.0
		switch {
		case p > 0:
			sign = 1
		case p < 0:
			sign = -1
		}
		gradient[i] = exponent * sign * math.Pow(math.Abs(p), exponent-1)
	}

	return gradient
}

func levyN13NDEvaluate(point []float64) float64 {

	n := len(point)
	w := make([]float64, n)
	for i, p := range point {
		w[i] = 1 + (p-1)/4
	}

	sinPw0 := math.Sin(math.Pi * w[0])
	sum := sinPw0 * sinPw0

	for i := 0; i < n-1; i++ {
		sinTerm := math.Sin(math.Pi * w[i+1])
		sum += (w[i] - 1) * (w[i] - 1) * (1 + 10*sinTerm*sinTerm)
	}

	sin2Pwn := math.Sin(2 * math.Pi * w[n-1])
	sum += (w[n-1] - 1) * (w[n-1] - 1) * (1 + sin2Pwn*sin2Pwn)

	return sum
}

func levyN13NDGradient(point []float64) []float64 {
	return NumericalGradient(levyN13NDEvaluate, point)
}

func ackleyN2Evaluate(point []float64) float64 {

	x, y := point[0], point[1]
	r := math.Sqrt(x*x + y*y)

	return -200 * math.Exp(-0.02*r)
}

func ackleyN2Gradient(point []float64) []float64 {

	x, y := point[0], point[1]
	r := math.Sqrt(x*x + y*y)

	if r <= 1e-10 {
		return []float64{0, 0}
	}

	factor := 200 * 0.02 * math.Exp(-0.02*r) / r

	return []float64{factor * x, factor * y}
}

func ackleyN3Evaluate(point []float64) float64 {

	x, y := point[0], point[1]
	r := math.Sqrt(x*x + y*y)

	return -200*math.Exp(-0.02*r) + 5*math.Exp(math.Cos(3*x)+math.Sin(3*y))
}

func ackleyN3Gradient(point []float64) []float64 {
	return NumericalGradient(ackleyN3Evaluate, point)
}

const xinSheYangN1Epsilon = 0.5

func xinSheYangN1Evaluate(point []float64) float64 {

	sum := 0.0
	for i, p := range point {
		sum += xinSheYangN1Epsilon * math.Pow(math.Abs(p), float64(i+1))
	}

	return sum
}

func xinSheYangN1Gradient(point []float64) []float64 {

	gradient := make([]float64, len(point))
	for i, p := range point {
		sign := 0.0
		switch {
		case p > 0:
			sign = 1
		case p < 0:
			sign = -1
		}
		exponent := float64(i + 1)
		gradient[i] = xinSheYangN1Epsilon * exponent * sign * math.Pow(math.Abs(p), exponent-1)
	}

	return gradient
}

func xinSheYangN2Evaluate(point []float64) float64 {

	sumAbs := 0.0
	sumSin := 0.0
	for _, p := range point {
		sumAbs += math.Abs(p)
		sumSin += math.Sin(p * p)
	}

	return sumAbs * math.Exp(-sumSin)
}

func xinSheYangN2Gradient(point []float64) []float64 {
	return NumericalGradient(xinSheYangN2Evaluate, point)
}

// Functions - registar poznatih funkcija koje se mogu koristiti za testiranje
var Functions = map[string]BenchmarkFunction{
	"sphere": {
		Name:        "sphere",
		Evaluate:    sphereEvaluate,
		Gradient:    sphereGradient,
		MinDim:      0,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"rosenbrock": {
		Name:        "rosenbrock",
		Evaluate:    rosenbrockEvaluate,
		Gradient:    rosenbrockGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[1,...,1]",
	},
	"rastrigin": {
		Name:        "rastrigin",
		Evaluate:    rastriginEvaluate,
		Gradient:    rastriginGradient,
		MinDim:      0,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"ackley": {
		Name:        "ackley",
		Evaluate:    ackleyEvaluate,
		Gradient:    ackleyGradient,
		MinDim:      0,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"himmelblau": {
		Name:        "himmelblau",
		Evaluate:    himmelblauEvaluate,
		Gradient:    himmelblauGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "[3,2], [-2.805,3.131], [-3.779,-3.283], [3.584,-1.848]",
	},
	"beale": {
		Name:        "beale",
		Evaluate:    bealeEvaluate,
		Gradient:    bealeGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "[3, 0.5]",
	},
	"booth": {
		Name:        "booth",
		Evaluate:    boothEvaluate,
		Gradient:    boothGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "[1, 3]",
	},
	"matyas": {
		Name:        "matyas",
		Evaluate:    matyasEvaluate,
		Gradient:    matyasGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "[0, 0]",
	},
	"levi": {
		Name:        "levi",
		Evaluate:    leviEvaluate,
		Gradient:    leviGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "[1, 1]",
	},
	"schwefel": {
		Name:        "schwefel",
		Evaluate:    schwefelEvaluate,
		Gradient:    schwefelGradient,
		MinDim:      0,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[420.9687,...,420.9687]",
	},
	"griewank": {
		Name:        "griewank",
		Evaluate:    griewankEvaluate,
		Gradient:    griewankGradient,
		MinDim:      0,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"styblinski_tang": {
		Name:        "styblinski_tang",
		Evaluate:    styblinskiTangEvaluate,
		Gradient:    styblinskiTangGradient,
		MinDim:      0,
		MaxDim:      0,
		GlobalMin:   0, // zavisi od n: -39.16599*n
		GlobalMinAt: "[-2.903534,...,-2.903534]",
	},
	"easom": {
		Name:        "easom",
		Evaluate:    easomEvaluate,
		Gradient:    easomGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -1,
		GlobalMinAt: "[π, π]",
	},
	"cross_in_tray": {
		Name:        "cross_in_tray",
		Evaluate:    crossInTrayEvaluate,
		Gradient:    crossInTrayGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -2.06261,
		GlobalMinAt: "[±1.3491, ±1.3491]",
	},
	"eggholder": {
		Name:        "eggholder",
		Evaluate:    eggholderEvaluate,
		Gradient:    eggholderGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -959.6407,
		GlobalMinAt: "[512, 404.2319]",
	},
	"bukin_n6": {
		Name:        "bukin_n6",
		Evaluate:    bukinN6Evaluate,
		Gradient:    bukinN6Gradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "[-10, 1]",
	},
	"drop_wave": {
		Name:        "drop_wave",
		Evaluate:    dropWaveEvaluate,
		Gradient:    dropWaveGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -1,
		GlobalMinAt: "[0, 0]",
	},
	"schaffer_n2": {
		Name:        "schaffer_n2",
		Evaluate:    schafferN2Evaluate,
		Gradient:    schafferN2Gradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "[0, 0]",
	},
	"six_hump_camel": {
		Name:        "six_hump_camel",
		Evaluate:    sixHumpCamelEvaluate,
		Gradient:    sixHumpCamelGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -1.0316,
		GlobalMinAt: "(-0.0898, 0.7126) i (0.0898, -0.7126)",
	},
	"three_hump_camel": {
		Name:        "three_hump_camel",
		Evaluate:    threeHumpCamelEvaluate,
		Gradient:    threeHumpCamelGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "(0, 0)",
	},
	"mccormick": {
		Name:        "mccormick",
		Evaluate:    mccormickEvaluate,
		Gradient:    mccormickGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -1.9133,
		GlobalMinAt: "(-0.5472, -1.5472)",
	},
	"shubert": {
		Name:        "shubert",
		Evaluate:    shubertEvaluate,
		Gradient:    shubertGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -186.7309,
		GlobalMinAt: "18 globalnih minimuma",
	},
	"michalewicz": {
		Name:        "michalewicz",
		Evaluate:    michalewiczEvaluate,
		Gradient:    michalewiczGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   -1.8013,
		GlobalMinAt: "zavisi od dimenzije",
	},
	"zakharov": {
		Name:        "zakharov",
		Evaluate:    zakharovEvaluate,
		Gradient:    zakharovGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"dixon_price": {
		Name:        "dixon_price",
		Evaluate:    dixonPriceEvaluate,
		Gradient:    dixonPriceGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "x_i = 2^(-(2^i-2)/2^i)",
	},
	"powell": {
		Name:        "powell",
		Evaluate:    powellEvaluate,
		Gradient:    powellGradient,
		MinDim:      4,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"rotated_hyper_ellipsoid": {
		Name:        "rotated_hyper_ellipsoid",
		Evaluate:    rotatedHyperEllipsoidEvaluate,
		Gradient:    rotatedHyperEllipsoidGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"sum_of_different_powers": {
		Name:        "sum_of_different_powers",
		Evaluate:    sumOfDifferentPowersEvaluate,
		Gradient:    sumOfDifferentPowersGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"trid": {
		Name:        "trid",
		Evaluate:    tridEvaluate,
		Gradient:    tridGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0, // zavisi od n: -n*(n+4)*(n-1)/6
		GlobalMinAt: "x_i = i*(n+1-i)",
	},
	"colville": {
		Name:        "colville",
		Evaluate:    colvilleEvaluate,
		Gradient:    colvilleGradient,
		MinDim:      4,
		MaxDim:      4,
		GlobalMin:   0,
		GlobalMinAt: "(1, 1, 1, 1)",
	},
	"perm": {
		Name:        "perm",
		Evaluate:    permEvaluate,
		Gradient:    permGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[1, 2, 3, ..., n]",
	},
	"levy_n13": {
		Name:        "levy_n13",
		Evaluate:    leviEvaluate,
		Gradient:    leviGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "(1, 1)",
	},
	"goldstein_price": {
		Name:        "goldstein_price",
		Evaluate:    goldsteinPriceEvaluate,
		Gradient:    goldsteinPriceGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   3,
		GlobalMinAt: "(0, -1)",
	},
	"bohachevsky_n1": {
		Name:        "bohachevsky_n1",
		Evaluate:    bohachevskyN1Evaluate,
		Gradient:    bohachevskyN1Gradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "(0, 0)",
	},
	"bohachevsky_n2": {
		Name:        "bohachevsky_n2",
		Evaluate:    bohachevskyN2Evaluate,
		Gradient:    bohachevskyN2Gradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "(0, 0)",
	},
	"bohachevsky_n3": {
		Name:        "bohachevsky_n3",
		Evaluate:    bohachevskyN3Evaluate,
		Gradient:    bohachevskyN3Gradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0,
		GlobalMinAt: "(0, 0)",
	},
	"schaffer_n4": {
		Name:        "schaffer_n4",
		Evaluate:    schafferN4Evaluate,
		Gradient:    schafferN4Gradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   0.292579,
		GlobalMinAt: "(0, 1.25313) i varijante",
	},
	"zettl": {
		Name:        "zettl",
		Evaluate:    zettlEvaluate,
		Gradient:    zettlGradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -0.003791,
		GlobalMinAt: "(-0.0299, 0)",
	},
	"shekel": {
		Name:        "shekel",
		Evaluate:    shekelEvaluate,
		Gradient:    shekelGradient,
		MinDim:      4,
		MaxDim:      4,
		GlobalMin:   -10.5364,
		GlobalMinAt: "(4, 4, 4, 4)",
	},
	"hartmann_3d": {
		Name:        "hartmann_3d",
		Evaluate:    hartmann3DEvaluate,
		Gradient:    hartmann3DGradient,
		MinDim:      3,
		MaxDim:      3,
		GlobalMin:   -3.8628,
		GlobalMinAt: "(0.1140, 0.5556, 0.8525)",
	},
	"hartmann_6d": {
		Name:        "hartmann_6d",
		Evaluate:    hartmann6DEvaluate,
		Gradient:    hartmann6DGradient,
		MinDim:      6,
		MaxDim:      6,
		GlobalMin:   -3.3224,
		GlobalMinAt: "(0.2017, 0.1500, 0.4769, 0.2753, 0.3117, 0.6573)",
	},
	"bent_cigar": {
		Name:        "bent_cigar",
		Evaluate:    bentCigarEvaluate,
		Gradient:    bentCigarGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"weierstrass": {
		Name:        "weierstrass",
		Evaluate:    weierstrassEvaluate,
		Gradient:    weierstrassGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"salomon": {
		Name:        "salomon",
		Evaluate:    salomonEvaluate,
		Gradient:    salomonGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"alpine_n1": {
		Name:        "alpine_n1",
		Evaluate:    alpineN1Evaluate,
		Gradient:    alpineN1Gradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"alpine_n2": {
		Name:        "alpine_n2",
		Evaluate:    alpineN2Evaluate,
		Gradient:    alpineN2Gradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0, // zavisi od n: priblizno -2.808^n
		GlobalMinAt: "[7.917,...,7.917]",
	},
	"periodic": {
		Name:        "periodic",
		Evaluate:    periodicEvaluate,
		Gradient:    periodicGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0.9,
		GlobalMinAt: "[0,...,0]",
	},
	"qing": {
		Name:        "qing",
		Evaluate:    qingEvaluate,
		Gradient:    qingGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "x_i = sqrt(i+1)",
	},
	"katsuura": {
		Name:        "katsuura",
		Evaluate:    katsuuraEvaluate,
		Gradient:    katsuuraGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"happycat": {
		Name:        "happycat",
		Evaluate:    happycatEvaluate,
		Gradient:    happycatGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[-1,...,-1]",
	},
	"hgbat": {
		Name:        "hgbat",
		Evaluate:    hgbatEvaluate,
		Gradient:    hgbatGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[-1,...,-1]",
	},
	"schwefel_221": {
		Name:        "schwefel_221",
		Evaluate:    schwefel221Evaluate,
		Gradient:    schwefel221Gradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"schwefel_222": {
		Name:        "schwefel_222",
		Evaluate:    schwefel222Evaluate,
		Gradient:    schwefel222Gradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"different_powers": {
		Name:        "different_powers",
		Evaluate:    differentPowersEvaluate,
		Gradient:    differentPowersGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"levy_n13_nd": {
		Name:        "levy_n13_nd",
		Evaluate:    levyN13NDEvaluate,
		Gradient:    levyN13NDGradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[1,...,1]",
	},
	"ackley_n2": {
		Name:        "ackley_n2",
		Evaluate:    ackleyN2Evaluate,
		Gradient:    ackleyN2Gradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -200,
		GlobalMinAt: "(0, 0)",
	},
	"ackley_n3": {
		Name:        "ackley_n3",
		Evaluate:    ackleyN3Evaluate,
		Gradient:    ackleyN3Gradient,
		MinDim:      2,
		MaxDim:      2,
		GlobalMin:   -195.629,
		GlobalMinAt: "(0.682, -0.36)",
	},
	"xin_she_yang_n1": {
		Name:        "xin_she_yang_n1",
		Evaluate:    xinSheYangN1Evaluate,
		Gradient:    xinSheYangN1Gradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
	"xin_she_yang_n2": {
		Name:        "xin_she_yang_n2",
		Evaluate:    xinSheYangN2Evaluate,
		Gradient:    xinSheYangN2Gradient,
		MinDim:      2,
		MaxDim:      0,
		GlobalMin:   0,
		GlobalMinAt: "[0,...,0]",
	},
}

// Get pronalazi benchmark funkciju po nazivu, sa podrazumevanom vrednošću "sphere" kada je name prazan string
func Get(name string) (BenchmarkFunction, error) {

	if name == "" {
		name = "sphere"
	}

	fn, ok := Functions[name]
	if !ok {
		return BenchmarkFunction{}, fmt.Errorf("unknown function: %s", name)
	}

	return fn, nil
}
