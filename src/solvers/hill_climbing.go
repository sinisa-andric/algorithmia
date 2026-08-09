package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	hillClimbingStepSize  = 0.1
	hillClimbingMaxSteps  = 1000
	hillClimbingTolerance = 1e-6
)

// HillClimbing minimizuje sphere funkciju koristeći hill climbing: isprobava korak od +/- step_size u svakoj
// dimenziji i zadržava svako poboljšanje
// problem.Point je početna tačka pretrage
func HillClimbing(problem models.Problem) (result models.Result, err error) {

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

	stepSize := hillClimbingStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}

	maxSteps := hillClimbingMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := hillClimbingTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	value := fn.Evaluate(point)

	steps := 0
	for ; steps < maxSteps; steps++ {

		previousValue := value

		for i := range point {
			for _, delta := range [2]float64{stepSize, -stepSize} {
				candidate := append([]float64(nil), point...)
				candidate[i] += delta

				candidateValue := fn.Evaluate(candidate)
				if candidateValue < value {
					point = candidate
					value = candidateValue
				}
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, value, false)
		}

		if previousValue-value < tolerance {
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, value, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "hill_climbing",
		Point:      point,
		Value:      value,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
