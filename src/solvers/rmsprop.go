package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	rmspropLearningRate = 0.001
	rmspropDecay        = 0.9
	rmspropEpsilon      = 1e-8
	rmspropMaxSteps     = 1000
	rmspropTolerance    = 1e-6
)

// Rmsprop minimizuje konfigurisanu benchmark funkciju koristeći RMSProp: learning rate po dimenziji se deli
// eksponencijalnim pokretnim prosekom kvadrata gradijenata
// problem.Point je početna tačka pretrage
func Rmsprop(problem models.Problem) (result models.Result, err error) {

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

	learningRate := rmspropLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	decay := rmspropDecay
	if v, ok := problem.Payload["decay"].(float64); ok {
		decay = v
	}

	epsilon := rmspropEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := rmspropMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := rmspropTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	meanSquare := make([]float64, len(point))
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}
		for i := range point {
			meanSquare[i] = decay*meanSquare[i] + (1-decay)*gradient[i]*gradient[i]
			point[i] -= learningRate * gradient[i] / (math.Sqrt(meanSquare[i]) + epsilon)
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
		Method:     "rmsprop",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
