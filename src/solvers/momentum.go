package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	momentumLearningRate = 0.01
	momentumMomentum     = 0.9
	momentumMaxSteps     = 1000
	momentumTolerance    = 1e-6
)

// Momentum minimizuje konfigurisanu benchmark funkciju koristeći gradijentni spust sa klasičnim momentumom: brzina
// akumulira eksponencijalni prosek prošlih gradijenata i pokreće ažuriranje pozicije
// problem.Point je početna tačka pretrage
func Momentum(problem models.Problem) (result models.Result, err error) {

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

	learningRate := momentumLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	momentumCoef := momentumMomentum
	if v, ok := problem.Payload["momentum"].(float64); ok {
		momentumCoef = v
	}

	maxSteps := momentumMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := momentumTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	velocity := make([]float64, len(point))
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}
		for i := range point {
			velocity[i] = momentumCoef*velocity[i] + learningRate*gradient[i]
			point[i] -= velocity[i]
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
		Method:     "momentum",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
