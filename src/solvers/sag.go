package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	sagLearningRate = 0.1
	sagMemorySize   = 5
	sagMaxSteps     = 1000
	sagTolerance    = 1e-6
	sagStepClip     = 1.0
)

// Sag minimizuje konfigurisanu benchmark funkciju koristeći Stochastic Average Gradient: drži poslednjih
// memory_size gradijenata u kružnom baferu i pomera se duž njihovog PRAVOG aritmetičkog proseka, za razliku od
// momentum.go koji koristi eksponencijalni pokretni prosek bez fiksnog broja članova
// problem.Point je početna tačka pretrage
func Sag(problem models.Problem) (result models.Result, err error) {

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

	learningRate := sagLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	memorySize := sagMemorySize
	if v, ok := problem.Payload["memory_size"].(float64); ok {
		memorySize = int(v)
	}
	if memorySize < 1 {
		memorySize = 1
	}

	maxSteps := sagMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sagTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	memory := make([][]float64, 0, memorySize)
	slot := 0

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		if len(memory) < memorySize {
			memory = append(memory, grad)
		} else {
			memory[slot%memorySize] = grad
		}
		slot++

		avg := make([]float64, dimensions)
		for _, g := range memory {
			for d := range avg {
				avg[d] += g[d]
			}
		}
		for d := range avg {
			avg[d] /= float64(len(memory))
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = -learningRate * avg[d]
		}
		delta = clipStep(delta, sagStepClip)
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
		Method:     "sag",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
