package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	adaMomentumLearningRate = 0.1
	adaMomentumBeta1        = 0.9
	adaMomentumBeta2        = 0.999
	adaMomentumEpsilon      = 1e-8
	adaMomentumMaxSteps     = 1000
	adaMomentumTolerance    = 1e-6
	adaMomentumStepClip     = 1.0
)

// AdaMomentum minimizuje konfigurisanu benchmark funkciju koristeći AdaMomentum optimizator: fiksni koeficijent
// momentuma beta1 se PRIGUŠUJE (skalira) drugim momentom pre upotrebe, za razliku od adai.go koji beta1 IZRAČUNAVA
// iznova iz odnosa prema proseku — ovde se postojeći hiperparametar beta1 samo modulira, ne zamenjuje potpuno
// izvedenom vrednošću
// problem.Point je početna tačka pretrage
func AdaMomentum(problem models.Problem) (result models.Result, err error) {

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

	learningRate := adaMomentumLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := adaMomentumBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := adaMomentumBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := adaMomentumEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := adaMomentumMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := adaMomentumTolerance
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
		delta := make([]float64, dimensions)
		for d := range delta {
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
			sqrtS := math.Sqrt(s[d])
			beta1Eff := beta1 * (sqrtS / (sqrtS + epsilon))
			m[d] = beta1Eff*m[d] + (1-beta1Eff)*grad[d]
			delta[d] = -learningRate * m[d] / (sqrtS + epsilon)
		}
		delta = clipStep(delta, adaMomentumStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
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
		Method:     "ada_momentum",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
