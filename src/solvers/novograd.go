package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	novogradLearningRate = 0.1
	novogradBeta1        = 0.95
	novogradBeta2        = 0.98
	novogradEpsilon      = 1e-8
	novogradWeightDecay  = 0.01
	novogradMaxSteps     = 1000
	novogradTolerance    = 1e-6
	novogradStepClip     = 1.0
)

// Novograd minimizuje konfigurisanu benchmark funkciju koristeći NovoGrad: drugi moment je JEDAN SKALAR za ceo
// gradijent (EMA njegove norme na kvadrat), ne vektor po dimenzijama kao Adam, a momentum uključuje i decoupled
// weight decay direktno u svojoj rekurziji
// problem.Point je početna tačka pretrage
func Novograd(problem models.Problem) (result models.Result, err error) {

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

	learningRate := novogradLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := novogradBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := novogradBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := novogradEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	weightDecay := novogradWeightDecay
	if v, ok := problem.Payload["weight_decay"].(float64); ok {
		weightDecay = v
	}

	maxSteps := novogradMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := novogradTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	s := 0.0

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		gradNormSq := 0.0
		for _, g := range grad {
			gradNormSq += g * g
		}
		s = beta2*s + (1-beta2)*gradNormSq

		delta := make([]float64, dimensions)
		for d := range delta {
			m[d] = beta1*m[d] + grad[d]/(math.Sqrt(s)+epsilon) + weightDecay*x[d]
			delta[d] = -learningRate * m[d]
		}
		delta = clipStep(delta, novogradStepClip)
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
		Method:     "novograd",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
