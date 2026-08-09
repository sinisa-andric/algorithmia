package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	sm3LearningRate = 0.1
	sm3Epsilon      = 1e-8
	sm3MaxSteps     = 1000
	sm3Tolerance    = 1e-6
	sm3StepClip     = 1.0
	sm3DecayEvery   = 100
	sm3DecayFactor  = 0.9
)

// Sm3 minimizuje konfigurisanu benchmark funkciju koristeći SM3 optimizator: prost kumulativni zbir kvadrata
// gradijenata po dimenziji (ne EMA kao rmsprop.go, ne monotono rastući bez opadanja kao adagrad.go) sa periodičnim
// opadanjem akumulatora svakih 100 koraka koje sprečava trajno gašenje efektivnog learning rate-a
// problem.Point je početna tačka pretrage
func Sm3(problem models.Problem) (result models.Result, err error) {

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

	learningRate := sm3LearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	epsilon := sm3Epsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := sm3MaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sm3Tolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	accumulator := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		if steps > 0 && steps%sm3DecayEvery == 0 {
			for d := range accumulator {
				accumulator[d] *= sm3DecayFactor
			}
		}

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			accumulator[d] += grad[d] * grad[d]
			delta[d] = -learningRate * grad[d] / (math.Sqrt(accumulator[d]) + epsilon)
		}
		delta = clipStep(delta, sm3StepClip)
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
		Method:     "sm3",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
