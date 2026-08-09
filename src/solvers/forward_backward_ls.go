package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	forwardBackwardLsMaxSteps        = 1000
	forwardBackwardLsLearningRate    = 0.1
	forwardBackwardLsLambda          = 0.01
	forwardBackwardLsBacktrackFactor = 0.5
	forwardBackwardLsTolerance       = 1e-6
	forwardBackwardLsStepClip        = 1.0
	forwardBackwardLsMaxBacktrack    = 10
)

// ForwardBackwardLs minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Forward-Backward sa
// backtracking linijskom pretragom: adaptivna veličina koraka koja se smanjuje dok se ne postigne dovoljan pad, za
// razliku od proximal_gradient.go koji koristi fiksan learning_rate kroz ceo run
// problem.Point je početna tačka pretrage
func ForwardBackwardLs(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := forwardBackwardLsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := forwardBackwardLsLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := forwardBackwardLsLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	backtrackFactor := forwardBackwardLsBacktrackFactor
	if v, ok := problem.Payload["backtrack_factor"].(float64); ok {
		backtrackFactor = v
	}

	tolerance := forwardBackwardLsTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	step := learningRate

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		xValue := objective(x)
		candidate := x
		for range forwardBackwardLsMaxBacktrack {
			raw := make([]float64, dimensions)
			for d := range raw {
				raw[d] = x[d] - step*grad[d]
			}
			candidate = proxL1(raw, step*lambda)
			if objective(candidate) <= xValue {
				break
			}
			step *= backtrackFactor
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = candidate[d] - x[d]
		}
		delta = clipStep(delta, forwardBackwardLsStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		step = math.Min(step*1.1, learningRate)

		value := objective(x)
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
		Method:     "forward_backward_ls",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
