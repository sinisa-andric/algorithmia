package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	adaiLearningRate = 0.1
	adaiBeta0        = 0.1
	adaiBeta2        = 0.999
	adaiEpsilon      = 1e-8
	adaiMaxSteps     = 1000
	adaiTolerance    = 1e-6
	adaiStepClip     = 1.0
)

// Adai minimizuje konfigurisanu benchmark funkciju koristeći Adai optimizator: koeficijent momentuma beta1 se NE
// zadaje fiksno kao hiperparametar, već se svaki korak IZRAČUNAVA iz odnosa drugog momenta dimenzije prema
// njegovom proseku — jedinstveno u ovom modulu
// problem.Point je početna tačka pretrage
func Adai(problem models.Problem) (result models.Result, err error) {

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

	learningRate := adaiLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta0 := adaiBeta0
	if v, ok := problem.Payload["beta0"].(float64); ok {
		beta0 = v
	}

	beta2 := adaiBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := adaiEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := adaiMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := adaiTolerance
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
		for d := range s {
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
		}

		meanS := 0.0
		for _, sv := range s {
			meanS += sv
		}
		meanS /= float64(dimensions)

		delta := make([]float64, dimensions)
		for d := range delta {
			beta1Adaptive := clamp(1-beta0*(s[d]/(meanS+epsilon)), 0, 0.9999)
			m[d] = beta1Adaptive*m[d] + (1-beta1Adaptive)*grad[d]
			delta[d] = -learningRate * m[d]
		}
		delta = clipStep(delta, adaiStepClip)
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
		Method:     "adai",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
