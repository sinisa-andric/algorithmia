package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	proximalGradientLearningRate = 0.1
	proximalGradientLambda       = 0.01
	proximalGradientMaxSteps     = 1000
	proximalGradientTolerance    = 1e-6
)

// ProximalGradient minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći proksimalni gradijentni metod: p
// ravi gradijentni korak na glavnom članu, a zatim primenjuje soft-thresholding za L1 član
// problem.Point je početna tačka pretrage
func ProximalGradient(problem models.Problem) (result models.Result, err error) {

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

	learningRate := proximalGradientLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := proximalGradientLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	maxSteps := proximalGradientMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := proximalGradientTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	threshold := learningRate * lambda

	steps := 0
	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		for i := range point {
			y := point[i] - learningRate*gradient[i]

			magnitude := math.Abs(y) - threshold
			if magnitude < 0 {
				magnitude = 0
			}
			point[i] = math.Copysign(magnitude, y)
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
		Method:     "proximal_gradient",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
