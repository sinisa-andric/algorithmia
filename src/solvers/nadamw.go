package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	nadamwLearningRate = 0.1
	nadamwBeta1        = 0.9
	nadamwBeta2        = 0.999
	nadamwEpsilon      = 1e-8
	nadamwWeightDecay  = 0.01
	nadamwMaxSteps     = 1000
	nadamwTolerance    = 1e-6
	nadamwStepClip     = 1.0
)

// Nadamw minimizuje konfigurisanu benchmark funkciju koristeći NAdamW: Nesterov-lookahead momentum (kao nadam.go)
// kombinovan sa DECOUPLED weight decay primenjenim kao odvojen korak nakon glavnog ažuriranja — isti odnos kao
// adamw.go prema adam.go, ali primenjen na Nesterov-lookahead formulu umesto na običan Adam
// problem.Point je početna tačka pretrage
func Nadamw(problem models.Problem) (result models.Result, err error) {

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

	learningRate := nadamwLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := nadamwBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := nadamwBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := nadamwEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	weightDecay := nadamwWeightDecay
	if v, ok := problem.Payload["weight_decay"].(float64); ok {
		weightDecay = v
	}

	maxSteps := nadamwMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := nadamwTolerance
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
		t := float64(steps + 1)
		beta1T := math.Pow(beta1, t)
		beta2T := math.Pow(beta2, t)

		delta := make([]float64, dimensions)
		for d := range delta {
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]

			mHat := (beta1*m[d] + (1-beta1)*grad[d]) / (1 - beta1T)
			sHat := s[d] / (1 - beta2T)

			delta[d] = -learningRate * mHat / (math.Sqrt(sHat) + epsilon)
		}
		delta = clipStep(delta, nadamwStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		wdDelta := make([]float64, dimensions)
		for d := range wdDelta {
			wdDelta[d] = -learningRate * weightDecay * x[d]
		}
		wdDelta = clipStep(wdDelta, nadamwStepClip)
		for d := range x {
			x[d] += wdDelta[d]
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
		Method:     "nadamw",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
