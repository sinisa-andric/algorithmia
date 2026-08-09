package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	mirrorProxMaxSteps     = 1000
	mirrorProxLearningRate = 0.1
	mirrorProxLambda       = 0.01
	mirrorProxTolerance    = 1e-6
	mirrorProxStepClip     = 1.0
)

// MirrorProx minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Mirror-Prox (ekstragradijent): dva
// gradijentna koraka po iteraciji (predictor pa corrector), za razliku od postojećeg mirror_descent.go koji radi
// samo jedan korak po iteraciji
// problem.Point je početna tačka pretrage
func MirrorProx(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := mirrorProxMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := mirrorProxLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := mirrorProxLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := mirrorProxTolerance
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

		grad1 := numGrad(fn.Evaluate, x, proximalGradEps)
		raw1 := make([]float64, dimensions)
		for d := range raw1 {
			raw1[d] = x[d] - learningRate*grad1[d]
		}
		xHalf := proxL1(raw1, learningRate*lambda)

		grad2 := numGrad(fn.Evaluate, xHalf, proximalGradEps)
		raw2 := make([]float64, dimensions)
		for d := range raw2 {
			raw2[d] = x[d] - learningRate*grad2[d]
		}
		xNew := proxL1(raw2, learningRate*lambda)

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xNew[d] - x[d]
		}
		delta = clipStep(delta, mirrorProxStepClip)
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
		Method:     "mirror_prox",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
