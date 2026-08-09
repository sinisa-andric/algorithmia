package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	apolloLearningRate = 0.1
	apolloBeta         = 0.9
	apolloEpsilon      = 1e-4
	apolloMaxSteps     = 1000
	apolloTolerance    = 1e-6
	apolloStepClip     = 1.0
	apolloBDiagInit    = 1.0
)

// Apollo minimizuje konfigurisanu benchmark funkciju koristeći Apollo optimizator: dijagonalna aproksimacija
// Hesijana se gradi iz secant informacije (razlika gradijenata podeljena razlikom pozicija) uz EMA glačanje, za
// razliku od proximal_newton.go koji koristi direktnu numeričku drugu izvod bez ikakvog istorijskog glačanja
// problem.Point je početna tačka pretrage
func Apollo(problem models.Problem) (result models.Result, err error) {

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

	learningRate := apolloLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta := apolloBeta
	if v, ok := problem.Payload["beta"].(float64); ok {
		beta = v
	}

	epsilon := apolloEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := apolloMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := apolloTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	xPrev := append([]float64(nil), x...)
	gradPrev := make([]float64, dimensions)
	bDiag := make([]float64, dimensions)
	for d := range bDiag {
		bDiag[d] = apolloBDiagInit
	}

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		for d := range bDiag {
			dDiff := grad[d] - gradPrev[d]
			denom := x[d] - xPrev[d] + epsilon
			secant := math.Abs(dDiff / denom)
			bDiag[d] = beta*bDiag[d] + (1-beta)*secant
		}

		xPrev = append([]float64(nil), x...)
		gradPrev = grad

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = -learningRate * grad[d] / (bDiag[d] + epsilon)
		}
		delta = clipStep(delta, apolloStepClip)
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
		Method:     "apollo",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
