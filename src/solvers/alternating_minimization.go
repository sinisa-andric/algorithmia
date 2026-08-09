package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	alternatingMinimizationMaxSteps     = 1000
	alternatingMinimizationLearningRate = 0.1
	alternatingMinimizationLambda       = 0.01
	alternatingMinimizationTolerance    = 1e-6
	alternatingMinimizationStepClip     = 1.0
)

// AlternatingMinimization minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći naizmeničnu
// minimizaciju po parnim i neparnim dimenzijama, za razliku od palm.go koji deli na prvu/drugu polovinu — ovde je
// podela po paritetu indeksa, sitnija granularnost
// problem.Point je početna tačka pretrage
func AlternatingMinimization(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := alternatingMinimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := alternatingMinimizationLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := alternatingMinimizationLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := alternatingMinimizationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		for d := 0; d < dimensions; d += 2 {
			candidate := proxL1Scalar(x[d]-learningRate*grad[d], learningRate*lambda)
			x[d] += clamp(candidate-x[d], -alternatingMinimizationStepClip, alternatingMinimizationStepClip)
		}

		grad = numGrad(fn.Evaluate, x, proximalGradEps)
		for d := 1; d < dimensions; d += 2 {
			candidate := proxL1Scalar(x[d]-learningRate*grad[d], learningRate*lambda)
			x[d] += clamp(candidate-x[d], -alternatingMinimizationStepClip, alternatingMinimizationStepClip)
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
		Method:     "alternating_minimization",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
