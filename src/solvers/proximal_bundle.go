package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	proximalBundleMaxSteps     = 1000
	proximalBundleLearningRate = 0.1
	proximalBundleLambda       = 0.01
	proximalBundleBundleSize   = 5
	proximalBundleTolerance    = 1e-6
	proximalBundleStepClip     = 1.0
)

// ProximalBundle minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći proksimalni bundle metod:
// agregira poslednjih bundle_size gradijenata i primenjuje prosek kao proksimalno-gradijentni korak, za razliku od
// postojećeg bundle_method.go koji nema proksimalni L1 operator — čist cutting-plane agregat
// problem.Point je početna tačka pretrage
func ProximalBundle(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := proximalBundleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := proximalBundleLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := proximalBundleLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	bundleSize := proximalBundleBundleSize
	if v, ok := problem.Payload["bundle_size"].(float64); ok {
		bundleSize = int(v)
	}

	tolerance := proximalBundleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	bundle := make([][]float64, 0, bundleSize)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		bundle = append(bundle, grad)
		if len(bundle) > bundleSize {
			bundle = bundle[1:]
		}

		avgGrad := make([]float64, dimensions)
		for _, g := range bundle {
			for d := range avgGrad {
				avgGrad[d] += g[d]
			}
		}
		for d := range avgGrad {
			avgGrad[d] /= float64(len(bundle))
		}

		raw := make([]float64, dimensions)
		for d := range raw {
			raw[d] = x[d] - learningRate*avgGrad[d]
		}
		xNew := proxL1(raw, learningRate*lambda)

		if objective(xNew) > objective(x) {
			bundle = bundle[:0]
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xNew[d] - x[d]
		}
		delta = clipStep(delta, proximalBundleStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		value := objective(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			// bestValue prati objective (fn + L1 penal), dok rezultat vraća čist fn.Evaluate(best) — koristi se
			// isti izraz kao Value: ispod da bi putanja bila konzistentna sa finalnim rezultatom
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
		Method:     "proximal_bundle",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
