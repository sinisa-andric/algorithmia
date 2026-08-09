package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	adanwLearningRate = 0.1
	adanwBeta1        = 0.98
	adanwBeta2        = 0.92
	adanwBeta3        = 0.99
	adanwEpsilon      = 1e-8
	adanwWeightDecay  = 0.01
	adanwMaxSteps     = 1000
	adanwTolerance    = 1e-6
	adanwStepClip     = 1.0
)

// Adanw minimizuje konfigurisanu benchmark funkciju koristeći AdanW: isti Adan update kao adan.go (momentum m,
// EMA razlike gradijenata v, treći EMA n na kombinaciji), ali sa DECOUPLED weight decay primenjenim kao odvojen
// korak POSLE glavnog Adan ažuriranja — isti odnos kao adamw.go prema adam.go
// problem.Point je početna tačka pretrage
func Adanw(problem models.Problem) (result models.Result, err error) {

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

	learningRate := adanwLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := adanwBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := adanwBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	beta3 := adanwBeta3
	if v, ok := problem.Payload["beta3"].(float64); ok {
		beta3 = v
	}

	epsilon := adanwEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	weightDecay := adanwWeightDecay
	if v, ok := problem.Payload["weight_decay"].(float64); ok {
		weightDecay = v
	}

	maxSteps := adanwMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := adanwTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	v := make([]float64, dimensions)
	n := make([]float64, dimensions)
	gradPrev := make([]float64, dimensions)

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
			diff := grad[d] - gradPrev[d]
			v[d] = beta2*v[d] + (1-beta2)*diff
			inner := grad[d] + beta2*diff
			n[d] = beta3*n[d] + (1-beta3)*inner*inner
			delta[d] = -learningRate * (m[d] + beta2*v[d]) / (math.Sqrt(n[d]) + epsilon)
		}
		delta = clipStep(delta, adanwStepClip)
		for d := range x {
			x[d] += delta[d]
		}
		gradPrev = grad

		wdDelta := make([]float64, dimensions)
		for d := range wdDelta {
			wdDelta[d] = -learningRate * weightDecay * x[d]
		}
		wdDelta = clipStep(wdDelta, adanwStepClip)
		for d := range x {
			x[d] += wdDelta[d]
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
		Method:     "adanw",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
