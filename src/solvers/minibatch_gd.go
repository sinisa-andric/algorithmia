package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	minibatchGdLearningRate = 0.01
	minibatchGdBatchSize    = 10
	minibatchGdNoiseScale   = 0.01
	minibatchGdMaxSteps     = 1000
	minibatchGdTolerance    = 1e-6
)

// MinibatchGd minimizuje konfigurisanu benchmark funkciju koristeći simulirani mini-batch gradijentni spust: na pravi
// gradijent u svakom koraku dodaje Gausov šum skaliran veličinom batch-a
// problem.Point je početna tačka pretrage
func MinibatchGd(problem models.Problem) (result models.Result, err error) {

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

	learningRate := minibatchGdLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	batchSize := minibatchGdBatchSize
	if v, ok := problem.Payload["batch_size"].(float64); ok {
		batchSize = int(v)
	}

	noiseScale := minibatchGdNoiseScale
	if v, ok := problem.Payload["noise_scale"].(float64); ok {
		noiseScale = v
	}

	maxSteps := minibatchGdMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := minibatchGdTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
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
		noisy := append([]float64(nil), gradient...)
		for i := range noisy {
			noisy[i] += noiseScale * randNorm() / math.Sqrt(float64(batchSize))
		}
		for i := range point {
			point[i] -= learningRate * noisy[i]
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
		Method:     "minibatch_gd",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
