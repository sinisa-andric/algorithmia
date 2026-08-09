package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	qhadamLearningRate = 0.1
	qhadamBeta1        = 0.9
	qhadamBeta2        = 0.999
	qhadamNu1          = 0.7
	qhadamNu2          = 1.0
	qhadamEpsilon      = 1e-8
	qhadamMaxSteps     = 1000
	qhadamTolerance    = 1e-6
	qhadamStepClip     = 1.0
)

// Qhadam minimizuje konfigurisanu benchmark funkciju koristeći QHAdam (Quasi-Hyperbolic Adam): brojilac i
// imenilac koraka su kvazi-hiperbolične MEŠAVINE trenutnog gradijenta i njegovih EMA momenata (koeficijenti nu1,
// nu2), umesto čistih momenata kao kod Adam-a
// problem.Point je početna tačka pretrage
func Qhadam(problem models.Problem) (result models.Result, err error) {

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

	learningRate := qhadamLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := qhadamBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := qhadamBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	nu1 := qhadamNu1
	if v, ok := problem.Payload["nu1"].(float64); ok {
		nu1 = v
	}

	nu2 := qhadamNu2
	if v, ok := problem.Payload["nu2"].(float64); ok {
		nu2 = v
	}

	epsilon := qhadamEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := qhadamMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := qhadamTolerance
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
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]

			numerator := (1-nu1)*grad[d] + nu1*m[d]
			denominator := math.Sqrt((1-nu2)*grad[d]*grad[d]+nu2*s[d]) + epsilon
			delta[d] = -learningRate * numerator / denominator
		}
		delta = clipStep(delta, qhadamStepClip)
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
		Method:     "qhadam",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
