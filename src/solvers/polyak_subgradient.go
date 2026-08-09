package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	polyakSubgradientMaxSteps  = 1000
	polyakSubgradientTolerance = 1e-6
)

// PolyakSubgradient minimizuje konfigurisanu benchmark funkciju koristeći subgradijentni spust sa Polyak korakom:
// koristi poznatu optimalnu vrednost funkcije da odredi veličinu svakog koraka
// problem.Point je početna tačka pretrage
func PolyakSubgradient(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := polyakSubgradientMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := polyakSubgradientTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		gradientNorm := norm(gradient)
		if gradientNorm < tolerance {
			break
		}

		step := fn.Evaluate(point) / (gradientNorm * gradientNorm)
		for i := range point {
			point[i] -= step * gradient[i]
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
		Method:     "polyak_subgradient",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
