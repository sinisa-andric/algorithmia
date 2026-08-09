package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	asgdLearningRate = 0.01
	asgdLambda       = 1e-4
	asgdAlpha        = 0.75
	asgdT0           = 1e6
	asgdMaxSteps     = 1000
	asgdTolerance    = 1e-6
)

// Asgd minimizuje konfigurisanu benchmark funkciju koristeći Averaged Stochastic Gradient Descent: standardni SGD
// korak se dodatno usrednjava kroz iteracije čim broj koraka pređe t0, a vraća se to usrednjeno rešenje
// problem.Point je početna tačka pretrage
func Asgd(problem models.Problem) (result models.Result, err error) {

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

	learningRate := asgdLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := asgdLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	alpha := asgdAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	t0 := asgdT0
	if v, ok := problem.Payload["t0"].(float64); ok {
		t0 = v
	}

	maxSteps := asgdMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := asgdTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	average := append([]float64(nil), problem.Point...)
	mu := 1.0
	eta := learningRate
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		t := float64(steps + 1)

		for i := range point {
			point[i] -= eta * (gradient[i] + lambda*point[i])
		}

		if t > t0 {
			mu = 1 / math.Pow(math.Max(1, t-t0), alpha)
		}

		for i := range average {
			average[i] = mu*point[i] + (1-mu)*average[i]
		}

		eta = learningRate / math.Pow(1+lambda*learningRate*t, alpha)

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, average, fn.Evaluate(average), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, average, fn.Evaluate(average), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "asgd",
		Point:      average,
		Value:      fn.Evaluate(average),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
