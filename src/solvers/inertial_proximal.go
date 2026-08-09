package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	inertialProximalMaxSteps     = 1000
	inertialProximalLearningRate = 0.1
	inertialProximalLambda       = 0.01
	inertialProximalAlpha        = 0.3
	inertialProximalTolerance    = 1e-6
	inertialProximalStepClip     = 1.0
)

// InertialProximal minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći inercijalni proksimalni
// algoritam: fiksni koeficijent momentuma alpha, za razliku od fista.go čiji momentum koeficijent raste i menja se
// svaki korak po specifičnoj formuli
// problem.Point je početna tačka pretrage
func InertialProximal(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := inertialProximalMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := inertialProximalLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := inertialProximalLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	alpha := inertialProximalAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	tolerance := inertialProximalTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	xPrev := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		y := make([]float64, dimensions)
		for d := range y {
			y[d] = x[d] + alpha*(x[d]-xPrev[d])
		}

		grad := numGrad(fn.Evaluate, y, proximalGradEps)
		raw := make([]float64, dimensions)
		for d := range raw {
			raw[d] = y[d] - learningRate*grad[d]
		}
		xProposed := proxL1(raw, learningRate*lambda)

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xProposed[d] - x[d]
		}
		delta = clipStep(delta, inertialProximalStepClip)

		xPrev = append([]float64(nil), x...)
		for d := range x {
			x[d] += delta[d]
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
		Method:     "inertial_proximal",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
