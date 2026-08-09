package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	neroLearningRate = 0.1
	neroBeta         = 0.999
	neroEpsilon      = 1e-8
	neroMaxSteps     = 1000
	neroTolerance    = 1e-6
	neroStepClip     = 1.0
)

// Nero minimizuje konfigurisanu benchmark funkciju koristeći Nero optimizator: kombinuje EMA normalizaciju drugog
// momenta norme gradijenta (skalar, kao novograd.go) I skaliranje normom pozicije (kao fromage.go) u jednoj
// formuli, mešavina oba pristupa
// problem.Point je početna tačka pretrage
func Nero(problem models.Problem) (result models.Result, err error) {

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

	learningRate := neroLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta := neroBeta
	if v, ok := problem.Payload["beta"].(float64); ok {
		beta = v
	}

	epsilon := neroEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := neroMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := neroTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	s := 0.0

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		gradNorm := norm(grad)
		s = beta*s + (1-beta)*gradNorm*gradNorm
		xNorm := norm(x)

		delta := make([]float64, dimensions)
		for d := range delta {
			normalized := grad[d] / (math.Sqrt(s) + epsilon)
			delta[d] = -learningRate * normalized * (xNorm / (gradNorm + epsilon))
		}
		delta = clipStep(delta, neroStepClip)
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
		Method:     "nero",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
