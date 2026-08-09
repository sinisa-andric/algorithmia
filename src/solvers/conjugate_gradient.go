package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	conjugateGradientMaxSteps  = 1000
	conjugateGradientTolerance = 1e-6
)

// ConjugateGradient minimizuje konfigurisanu benchmark funkciju koristeći Fletcher-Reeves konjugovani gradijent:
// pravac pretrage kombinuje trenutni gradijent sa prethodnim pravcem ponderisanim odnosom kvadrata normi gradijenata
// problem.Point je početna tačka pretrage
func ConjugateGradient(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	maxSteps := conjugateGradientMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := conjugateGradientTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	gradient := fn.Gradient(point)
	direction := negate(gradient)

	steps := 0
	for ; steps < maxSteps; steps++ {
		if norm(gradient) < tolerance {
			break
		}

		// Hesijan sphere funkcije je 2*I, pa je d^T*A*d jednako 2*dot(d,d).
		curvature := 2 * dot(direction, direction)
		if curvature == 0 {
			break
		}
		alpha := dot(gradient, gradient) / curvature

		for i := range point {
			point[i] += alpha * direction[i]
		}

		nextGradient := fn.Gradient(point)
		beta := dot(nextGradient, nextGradient) / dot(gradient, gradient)

		for i := range direction {
			direction[i] = -nextGradient[i] + beta*direction[i]
		}

		gradient = nextGradient

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "conjugate_gradient",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func negate(vector []float64) []float64 {

	result := make([]float64, len(vector))
	for i, v := range vector {
		result[i] = -v
	}

	return result
}

func dot(a, b []float64) float64 {

	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}
