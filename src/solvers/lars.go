package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	larsMomentumCoef = 0.9
	larsTrustCoef    = 0.001
	larsWeightDecay  = 0.0001
	larsMaxSteps     = 1000
	larsTolerance    = 1e-6
	larsStepClip     = 1.0
)

// Lars minimizuje konfigurisanu benchmark funkciju koristeći LARS: čist momentum SGD sa lokalnom (layer-wise) stopom
// izvedenom iz odnosa norme pozicije i norme gradijenta, za razliku od lamb.go koji ima adaptivni drugi moment kao
// Adam — LARS nema nikakav adaptivni drugi moment
// problem.Point je početna tačka pretrage
func Lars(problem models.Problem) (result models.Result, err error) {

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

	momentumCoef := larsMomentumCoef
	if v, ok := problem.Payload["momentum_coef"].(float64); ok {
		momentumCoef = v
	}

	trustCoef := larsTrustCoef
	if v, ok := problem.Payload["trust_coef"].(float64); ok {
		trustCoef = v
	}

	weightDecay := larsWeightDecay
	if v, ok := problem.Payload["weight_decay"].(float64); ok {
		weightDecay = v
	}

	maxSteps := larsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := larsTolerance
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

		delta := make([]float64, dimensions)
		for d := range delta {
			m[d] = momentumCoef*m[d] + localLr*(grad[d]+weightDecay*x[d])
			delta[d] = -m[d]
		}
		delta = clipStep(delta, larsStepClip)
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
		Method:     "lars",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
