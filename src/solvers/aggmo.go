package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	aggmoLearningRate = 0.1
	aggmoBeta0        = 0.0
	aggmoBeta1        = 0.9
	aggmoBeta2        = 0.99
	aggmoMaxSteps     = 1000
	aggmoTolerance    = 1e-6
	aggmoStepClip     = 1.0
)

// Aggmo minimizuje konfigurisanu benchmark funkciju koristeći Aggregated Momentum: drži TRI paralelna momentum
// bafera sa različitim vremenskim skalama (beta_0, beta_1, beta_2) i pomera se duž njihovog proseka — jedinstven
// multi-skala pristup, za razliku od bilo kog pojedinačnog EMA momentuma u ovom modulu
// problem.Point je početna tačka pretrage
func Aggmo(problem models.Problem) (result models.Result, err error) {

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

	learningRate := aggmoLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta0 := aggmoBeta0
	if v, ok := problem.Payload["beta_0"].(float64); ok {
		beta0 = v
	}

	beta1 := aggmoBeta1
	if v, ok := problem.Payload["beta_1"].(float64); ok {
		beta1 = v
	}

	beta2 := aggmoBeta2
	if v, ok := problem.Payload["beta_2"].(float64); ok {
		beta2 = v
	}

	maxSteps := aggmoMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := aggmoTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m0 := make([]float64, dimensions)
	m1 := make([]float64, dimensions)
	m2 := make([]float64, dimensions)

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
			m0[d] = beta0*m0[d] + grad[d]
			m1[d] = beta1*m1[d] + grad[d]
			m2[d] = beta2*m2[d] + grad[d]
			delta[d] = -(learningRate / 3) * (m0[d] + m1[d] + m2[d])
		}
		delta = clipStep(delta, aggmoStepClip)
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
		Method:     "aggmo",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
