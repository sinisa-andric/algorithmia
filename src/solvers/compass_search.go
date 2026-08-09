package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	compassSearchStepSize      = 1.0
	compassSearchStepReduction = 0.5
	compassSearchTolerance     = 1e-6
	compassSearchMaxSteps      = 1000
)

// CompassSearch minimizuje konfigurisanu benchmark funkciju koristeći compass (koordinatnu) pretragu: ispituje korak
// duž svake koordinatne ose i smanjuje veličinu koraka kad nijedan pravac ne poboljša rezultat
// problem.Point je početna tačka pretrage
func CompassSearch(problem models.Problem) (result models.Result, err error) {

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

	stepSize := compassSearchStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}

	stepReduction := compassSearchStepReduction
	if v, ok := problem.Payload["step_reduction"].(float64); ok {
		stepReduction = v
	}

	tolerance := compassSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	maxSteps := compassSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	n := len(point)
	steps := 0

	for ; steps < maxSteps; steps++ {
		value := fn.Evaluate(point)
		improved := false

		for i := 0; i < n; i++ {
			for _, direction := range [2]float64{1, -1} {
				candidate := append([]float64(nil), point...)
				candidate[i] += stepSize * direction

				if fn.Evaluate(candidate) < value {
					point = candidate
					improved = true
					break
				}
			}
		}

		if !improved {
			stepSize *= stepReduction
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}

		if stepSize < tolerance {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "compass_search",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
