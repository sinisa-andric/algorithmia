package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	lambLearningRate = 0.1
	lambBeta1        = 0.9
	lambBeta2        = 0.999
	lambEpsilon      = 1e-6
	lambWeightDecay  = 0.01
	lambMaxSteps     = 1000
	lambTolerance    = 1e-6
	lambStepClip     = 1.0
)

// Lamb minimizuje konfigurisanu benchmark funkciju koristeći LAMB: standardni AdamW korak se dodatno skalira
// trust_ratio-om (odnos norme pozicije i norme predloženog koraka), za razliku od adamw.go koji nema takvo
// skaliranje
// problem.Point je početna tačka pretrage
func Lamb(problem models.Problem) (result models.Result, err error) {

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

	learningRate := lambLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := lambBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := lambBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := lambEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	weightDecay := lambWeightDecay
	if v, ok := problem.Payload["weight_decay"].(float64); ok {
		weightDecay = v
	}

	maxSteps := lambMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := lambTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	s := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		adamStep := make([]float64, dimensions)
		for d := range adamStep {
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
			adamStep[d] = m[d]/(math.Sqrt(s[d])+epsilon) + weightDecay*x[d]
		}

		xNorm := norm(x)
		stepNorm := norm(adamStep)
		trustRatio := 1.0
		if xNorm >= epsilon {
			trustRatio = xNorm / (stepNorm + epsilon)
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = -learningRate * trustRatio * adamStep[d]
		}
		delta = clipStep(delta, lambStepClip)
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
		Method:     "lamb",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
