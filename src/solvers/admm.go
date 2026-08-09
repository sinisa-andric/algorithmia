package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	admmMaxSteps     = 1000
	admmLearningRate = 0.1
	admmLambda       = 0.01
	admmRho          = 1.0
	admmTolerance    = 1e-6
	admmStepClip     = 1.0
)

// Admm minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći ADMM: naizmenično rešava primarnu
// promenljivu x i pomoćnu z uz Lagranžijan penal rho i skaliranu dualnu promenljivu u koja ih povezuje
// problem.Point je početna tačka pretrage
func Admm(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := admmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := admmLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := admmLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	rho := admmRho
	if v, ok := problem.Payload["rho"].(float64); ok {
		rho = v
	}

	tolerance := admmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	z := append([]float64(nil), problem.Point...)
	u := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = -learningRate * (grad[d] + rho*(x[d]-z[d]+u[d]))
		}
		delta = clipStep(delta, admmStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		zRaw := make([]float64, dimensions)
		for d := range zRaw {
			zRaw[d] = x[d] + u[d]
		}
		z = proxL1(zRaw, lambda/(rho+1e-10))

		for d := range u {
			u[d] += x[d] - z[d]
		}

		value := objective(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "admm",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
