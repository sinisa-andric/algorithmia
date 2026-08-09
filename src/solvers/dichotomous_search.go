package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
)

const (
	dichotomousSearchA         = -10.0
	dichotomousSearchB         = 10.0
	dichotomousSearchEpsilon   = 1e-4
	dichotomousSearchTolerance = 1e-6
	dichotomousSearchMaxSteps  = 1000
)

// DichotomousSearch minimizuje konfigurisanu benchmark funkciju koristeći dihotomnu pretragu: u svakom koraku poredi
// dve tačke blizu sredine intervala i odbacuje polovinu koja ne sadrži minimum
// problem.Point određuje dimenzionalnost rezultata, ostale koordinate ostaju na nuli
func DichotomousSearch(problem models.Problem) (result models.Result, err error) {

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

	a := dichotomousSearchA
	if v, ok := problem.Payload["a"].(float64); ok {
		a = v
	}

	b := dichotomousSearchB
	if v, ok := problem.Payload["b"].(float64); ok {
		b = v
	}

	epsilon := dichotomousSearchEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	tolerance := dichotomousSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	maxSteps := dichotomousSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	eval := func(xVal float64) float64 {
		point := make([]float64, dims)
		point[0] = xVal
		return fn.Evaluate(point)
	}

	steps := 0
	for ; steps < maxSteps; steps++ {
		mid := (a + b) / 2
		x1 := mid - epsilon/2
		x2 := mid + epsilon/2
		f1 := eval(x1)
		f2 := eval(x2)

		if f1 < f2 {
			b = x2
		} else {
			a = x1
		}

		if includeTrajectory {
			midX := (a + b) / 2
			p := make([]float64, dims)
			p[0] = midX
			trajectory = recordTrajectory(trajectory, steps, p, eval(midX), false)
		}

		if b-a < tolerance {
			steps++
			break
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
		Method:     "dichotomous_search",
		Point:      point,
		Value:      eval(x),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
