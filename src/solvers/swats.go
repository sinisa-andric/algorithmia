package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	swatsLearningRate = 0.1
	swatsBeta1        = 0.9
	swatsBeta2        = 0.999
	swatsEpsilon      = 1e-8
	swatsMaxSteps     = 1000
	swatsTolerance    = 1e-6
	swatsStepClip     = 1.0
)

// Swats minimizuje konfigurisanu benchmark funkciju koristeći SWATS: prva polovina koraka ponaša se kao Adam
// (m,s EMA sa bias korekcijom), a zatim EKSPLICITNO prelazi na čist SGD sa momentumom koristeći NAUČENU
// efektivnu stopu usrednjenu iz Adam faze — razlika od svih ostalih solvera koji koriste jedan režim tokom celog
// izvršavanja
// problem.Point je početna tačka pretrage
func Swats(problem models.Problem) (result models.Result, err error) {

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

	learningRate := swatsLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := swatsBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := swatsBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := swatsEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := swatsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := swatsTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	phaseSwitch := maxSteps / 2

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	s := make([]float64, dimensions)
	momentumBuf := make([]float64, dimensions)

	effectiveLrSum := 0.0
	effectiveLrCount := 0
	avgEffectiveLr := learningRate
	avgComputed := false

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)

		if steps < phaseSwitch {
			t := float64(steps + 1)
			beta1T := math.Pow(beta1, t)
			beta2T := math.Pow(beta2, t)
			for d := range delta {
				m[d] = beta1*m[d] + (1-beta1)*grad[d]
				s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
				mHat := m[d] / (1 - beta1T)
				sHat := s[d] / (1 - beta2T)
				delta[d] = -learningRate * mHat / (math.Sqrt(sHat) + epsilon)
			}
			ratio := norm(delta) / (norm(grad) + epsilon)
			effectiveLrSum += ratio
			effectiveLrCount++
		} else {
			if !avgComputed {
				if effectiveLrCount > 0 {
					avgEffectiveLr = effectiveLrSum / float64(effectiveLrCount)
				}
				avgComputed = true
			}
			for d := range delta {
				momentumBuf[d] = beta1*momentumBuf[d] + (1-beta1)*grad[d]
				delta[d] = -avgEffectiveLr * momentumBuf[d]
			}
		}

		delta = clipStep(delta, swatsStepClip)
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
		Method:     "swats",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
