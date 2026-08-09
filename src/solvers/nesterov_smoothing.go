package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	nesterovSmoothingMaxSteps     = 1000
	nesterovSmoothingLearningRate = 0.1
	nesterovSmoothingLambda       = 0.01
	nesterovSmoothingMuInit       = 1.0
	nesterovSmoothingTolerance    = 1e-6
	nesterovSmoothingStepClip     = 1.0
)

// NesterovSmoothing minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Nesterovljevo zaglađivanje:
// L1 član se aproksimira glatkom Huber-like funkcijom sa parametrom mu koji opada tokom izvršavanja, bez ikakvog
// proksimalnog operatora — za razliku od svih ostalih solvera u modulu, ovaj radi čist gradijentni spust na
// zaglađenoj kompozitnoj funkciji
// problem.Point je početna tačka pretrage
func NesterovSmoothing(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := nesterovSmoothingMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := nesterovSmoothingLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := nesterovSmoothingLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	muInit := nesterovSmoothingMuInit
	if v, ok := problem.Payload["mu_init"].(float64); ok {
		muInit = v
	}

	tolerance := nesterovSmoothingTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		mu := muInit*(1-float64(steps)/float64(maxSteps)) + 0.01

		gradF := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			smoothL1Grad := lambda * x[d] / math.Max(math.Abs(x[d]), mu)
			delta[d] = -learningRate * (gradF[d] + smoothL1Grad)
		}
		delta = clipStep(delta, nesterovSmoothingStepClip)
		for d := range x {
			x[d] += delta[d]
		}

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
		Method:     "nesterov_smoothing",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
