package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	bundleMethodLearningRate = 0.1
	bundleMethodMaxSteps     = 1000
	bundleMethodTolerance    = 1e-6
	bundleMethodBundleSize   = 10
)

// BundleMethod minimizuje konfigurisanu benchmark funkciju koristeći bundle metod: drži skup nedavnih subgradijenata
// i spušta se duž njihovog proseka
// problem.Point je početna tačka pretrage
func BundleMethod(problem models.Problem) (result models.Result, err error) {

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

	learningRate := bundleMethodLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	maxSteps := bundleMethodMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := bundleMethodTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	bundleSize := bundleMethodBundleSize
	if v, ok := problem.Payload["bundle_size"].(float64); ok {
		bundleSize = int(v)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	dimensions := len(point)
	bundle := make([][]float64, 0, bundleSize)

	steps := 0
	for ; steps < maxSteps; steps++ {

		gradient := fn.Gradient(point)

		bundle = append(bundle, gradient)
		if len(bundle) > bundleSize {
			bundle = bundle[1:]
		}

		aggregate := make([]float64, dimensions)
		for _, g := range bundle {
			for i := range aggregate {
				aggregate[i] += g[i]
			}
		}
		for i := range aggregate {
			aggregate[i] /= float64(len(bundle))
		}

		if norm(aggregate) < tolerance {
			break
		}

		for i := range point {
			point[i] -= learningRate * aggregate[i]
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
		Method:     "bundle_method",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
