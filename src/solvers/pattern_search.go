package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	patternSearchStepSize  = 1.0
	patternSearchMaxSteps  = 1000
	patternSearchTolerance = 1e-6
)

// PatternSearch minimizuje sphere funkciju koristeći pattern search: ispituje +/- step_size duž svake ose i
// prepolovljuje step_size kad god nijedan pravac ne poboljša rezultat
// problem.Point je početna tačka pretrage
func PatternSearch(problem models.Problem) (result models.Result, err error) {

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

	stepSize := patternSearchStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}

	maxSteps := patternSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := patternSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	value := fn.Evaluate(point)

	steps := 0
	for ; steps < maxSteps && stepSize >= tolerance; steps++ {

		improved := false

		for i := range point {
			for _, delta := range [2]float64{stepSize, -stepSize} {
				candidate := append([]float64(nil), point...)
				candidate[i] += delta

				candidateValue := fn.Evaluate(candidate)
				if candidateValue < value {
					point = candidate
					value = candidateValue
					improved = true
				}
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, value, false)
		}

		if !improved {
			stepSize /= 2
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, value, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "pattern_search",
		Point:      point,
		Value:      value,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
