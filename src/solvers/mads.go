package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	madsMaxSteps  = 1000
	madsTolerance = 1e-6
	madsDelta     = 1.0
)

// MADS minimizuje sphere funkciju koristeći pojednostavljeni Mesh Adaptive Direct Search: ispituje duž svake
// koordinatne ose, udvostručujući veličinu mreže pri uspehu, a prepolovljujući je inače
// problem.Point je početna tačka pretrage
func MADS(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := madsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := madsTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	delta := madsDelta
	if v, ok := problem.Payload["delta"].(float64); ok {
		delta = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	value := fn.Evaluate(point)

	steps := 0
	for ; steps < maxSteps && delta >= tolerance; steps++ {

		improved := false
		var bestPoll []float64
		bestPollValue := value

		for i := range point {
			for _, sign := range [2]float64{1, -1} {
				poll := append([]float64(nil), point...)
				poll[i] += sign * delta

				if pollValue := fn.Evaluate(poll); pollValue < bestPollValue {
					bestPollValue = pollValue
					bestPoll = poll
					improved = true
				}
			}
		}

		if improved {
			point = bestPoll
			value = bestPollValue
			delta *= 2.0
		} else {
			delta *= 0.5
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, value, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, value, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "mads",
		Point:      point,
		Value:      value,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
