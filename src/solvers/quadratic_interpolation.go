package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	quadraticInterpolationStep      = 1.0
	quadraticInterpolationMaxSteps  = 1000
	quadraticInterpolationTolerance = 1e-6
)

// QuadraticInterpolation minimizuje sphere funkciju koristeći kvadratnu interpolaciju: za svaku dimenziju nezavisno
// provlači parabolu kroz (x-h, x, x+h) i pomera se na njen analitički minimum
// problem.Point je početna tačka pretrage
func QuadraticInterpolation(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := quadraticInterpolationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := quadraticInterpolationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	h := quadraticInterpolationStep
	point := append([]float64(nil), problem.Point...)
	steps := 0

	for ; steps < maxSteps; steps++ {
		if norm(fn.Gradient(point)) < tolerance {
			break
		}

		for i := range point {
			minus := append([]float64(nil), point...)
			minus[i] -= h
			plus := append([]float64(nil), point...)
			plus[i] += h

			fMinus := fn.Evaluate(minus)
			fPlus := fn.Evaluate(plus)
			fCenter := fn.Evaluate(point)

			denominator := fPlus - 2*fCenter + fMinus
			if denominator == 0 {
				continue
			}

			point[i] -= h * (fPlus - fMinus) / (2 * denominator)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "quadratic_interpolation",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
