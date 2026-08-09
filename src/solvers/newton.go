package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	newtonLearningRate = 0.5
	newtonMaxSteps     = 100
	newtonTolerance    = 1e-6
	newtonHessianStep  = 1e-5
)

// Newton minimizuje konfigurisanu benchmark funkciju koristeći Newton-ov metod: dijagonala Hesijana se aproksimira
// numerički, a korak po svakoj dimenziji je gradijent podeljen tom aproksimacijom
// problem.Point je početna tačka pretrage
func Newton(problem models.Problem) (result models.Result, err error) {

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

	learningRate := newtonLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	maxSteps := newtonMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := newtonTolerance
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

		center := fn.Evaluate(point)
		for i := range point {
			plus := append([]float64(nil), point...)
			minus := append([]float64(nil), point...)
			plus[i] += newtonHessianStep
			minus[i] -= newtonHessianStep

			secondDerivative := (fn.Evaluate(plus) - 2*center + fn.Evaluate(minus)) / (newtonHessianStep * newtonHessianStep)

			if math.Abs(secondDerivative) < 1e-12 {
				point[i] -= learningRate * gradient[i]
				continue
			}
			point[i] -= learningRate * gradient[i] / secondDerivative
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
		Method:     "newton",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
