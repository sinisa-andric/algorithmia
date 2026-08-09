package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	sgdLearningRate = 0.01
	sgdNoiseScale   = 0.01
	sgdMaxSteps     = 1000
	sgdTolerance    = 1e-6
)

// Sgd minimizuje konfigurisanu benchmark funkciju koristeći stohastički gradijentni spust: svaki korak gradijentu
// dodaje Gausov šum bez usrednjavanja po batch-u (za razliku od minibatch_gd), simulirajući uzorkovanje jednog primera
// problem.Point je početna tačka pretrage
func Sgd(problem models.Problem) (result models.Result, err error) {

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

	learningRate := sgdLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	maxSteps := sgdMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sgdTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	noiseScale := sgdNoiseScale
	if v, ok := problem.Payload["noise_scale"].(float64); ok {
		noiseScale = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}
		noisyGradient := append([]float64(nil), gradient...)
		for i := range noisyGradient {
			noisyGradient[i] += noiseScale * randNorm()
		}
		for i := range point {
			point[i] -= learningRate * noisyGradient[i]
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
		Method:     "sgd",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
