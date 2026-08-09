package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
)

const (
	fibonacciSearchA         = -10.0
	fibonacciSearchB         = 10.0
	fibonacciSearchN         = 20
	fibonacciSearchTolerance = 1e-6
)

// FibonacciSearch minimizuje konfigurisanu benchmark funkciju koristeći Fibonačijevu pretragu: koristi Fibonačijeve
// brojeve da postavi dve probne tačke po iteraciji i suzi interval ka minimumu
// problem.Point određuje dimenzionalnost rezultata, ostale koordinate ostaju na nuli
func FibonacciSearch(problem models.Problem) (result models.Result, err error) {

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}

	dims := len(problem.Point)
	if dims == 0 {
		dims = 1
	}
	if err := fn.ValidateDimension(dims); err != nil {
		return result, err
	}

	a := fibonacciSearchA
	if v, ok := problem.Payload["a"].(float64); ok {
		a = v
	}

	b := fibonacciSearchB
	if v, ok := problem.Payload["b"].(float64); ok {
		b = v
	}

	n := fibonacciSearchN
	if v, ok := problem.Payload["n"].(float64); ok {
		n = int(v)
	}
	if n < 2 {
		n = 2
	}

	eval := func(xVal float64) float64 {
		point := make([]float64, dims)
		point[0] = xVal
		return fn.Evaluate(point)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	fib := make([]float64, n+1)
	fib[0], fib[1] = 1, 1
	for i := 2; i <= n; i++ {
		fib[i] = fib[i-1] + fib[i-2]
	}

	steps := 0
	for k := n; k >= 2; k-- {
		x1 := a + fib[k-2]/fib[k]*(b-a)
		x2 := a + fib[k-1]/fib[k]*(b-a)
		f1 := eval(x1)
		f2 := eval(x2)

		if f1 > f2 {
			a = x1
		} else {
			b = x2
		}
		steps++

		if includeTrajectory {
			mid := (a + b) / 2
			midPoint := make([]float64, dims)
			midPoint[0] = mid
			trajectory = recordTrajectory(trajectory, steps, midPoint, eval(mid), false)
		}
	}

	x := (a + b) / 2
	point := make([]float64, dims)
	point[0] = x

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, eval(x), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "fibonacci_search",
		Point:      point,
		Value:      eval(x),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
