package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	randomSearchSamples = 10000
	randomSearchRange   = 15.0
)

// RandomSearch minimizuje sphere funkciju koristeći čisto Monte Karlo uzorkovanje: povlači nasumične tačke u [-range,
// range] po dimenziji i zadržava najbolju
// problem.Point samo određuje dimenzionalnost
func RandomSearch(problem models.Problem) (result models.Result, err error) {

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

	samples := randomSearchSamples
	if v, ok := problem.Payload["samples"].(float64); ok {
		samples = int(v)
	}

	searchRange := randomSearchRange
	if v, ok := problem.Payload["range"].(float64); ok {
		searchRange = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	best := make([]float64, dimensions)
	bestValue := math.Inf(1)

	steps := 0
	for ; steps < samples; steps++ {

		candidate := make([]float64, dimensions)
		for i := range candidate {
			candidate[i] = (rand.Float64()*2 - 1) * searchRange
		}

		candidateValue := fn.Evaluate(candidate)
		if candidateValue < bestValue {
			bestValue = candidateValue
			best = candidate
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "random_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
