package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	larcMomentumCoef = 0.9
	larcTrustCoef    = 0.001
	larcClipMax      = 1.0
	larcWeightDecay  = 0.0001
	larcMaxSteps     = 1000
	larcTolerance    = 1e-6
	larcStepClip     = 1.0
)

// Larc minimizuje konfigurisanu benchmark funkciju koristeći LARC: isto kao lars.go, ali sa dodatnim gornjim
// ograničenjem na izračunatu lokalnu (layer-wise) stopu — "clipped" verzija koja sprečava prevelike korake kada je
// norma gradijenta mala u odnosu na normu pozicije
// problem.Point je početna tačka pretrage
func Larc(problem models.Problem) (result models.Result, err error) {

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

	momentumCoef := larcMomentumCoef
	if v, ok := problem.Payload["momentum_coef"].(float64); ok {
		momentumCoef = v
	}

	trustCoef := larcTrustCoef
	if v, ok := problem.Payload["trust_coef"].(float64); ok {
		trustCoef = v
	}

	clipMax := larcClipMax
	if v, ok := problem.Payload["clip_max"].(float64); ok {
		clipMax = v
	}

	weightDecay := larcWeightDecay
	if v, ok := problem.Payload["weight_decay"].(float64); ok {
		weightDecay = v
	}

	maxSteps := larcMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := larcTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		xNorm := norm(x)
		gradNorm := norm(grad)
		localLr := trustCoef * xNorm / (gradNorm + weightDecay*xNorm + 1e-8)
		localLr = math.Min(localLr, clipMax)

		delta := make([]float64, dimensions)
		for d := range delta {
			m[d] = momentumCoef*m[d] + localLr*(grad[d]+weightDecay*x[d])
			delta[d] = -m[d]
		}
		delta = clipStep(delta, larcStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		value := fn.Evaluate(x)
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
		Method:     "larc",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
