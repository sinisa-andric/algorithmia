package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	nagLearningRate = 0.01
	nagMomentum     = 0.9
	nagMaxSteps     = 1000
	nagTolerance    = 1e-6
)

// Nag minimizuje konfigurisanu benchmark funkciju koristeći Nesterov Accelerated Gradient: gradijent se računa u
// tački pomerenoj unapred po pravcu tekuće brzine (lookahead), pre nego što se brzina i pozicija ažuriraju
// problem.Point je početna tačka pretrage
func Nag(problem models.Problem) (result models.Result, err error) {

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

	learningRate := nagLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	momentumCoef := nagMomentum
	if v, ok := problem.Payload["momentum"].(float64); ok {
		momentumCoef = v
	}

	maxSteps := nagMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := nagTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	velocity := make([]float64, len(point))
	steps := 0

	for ; steps < maxSteps; steps++ {
		ahead := make([]float64, len(point))
		for i := range point {
			ahead[i] = point[i] - momentumCoef*velocity[i]
		}

		gradient := fn.Gradient(ahead)
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
		Method:     "nag",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
