package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	bregmanProximalGradientMaxSteps     = 1000
	bregmanProximalGradientLearningRate = 0.1
	bregmanProximalGradientLambda       = 0.01
	bregmanProximalGradientTolerance    = 1e-6
	bregmanProximalGradientStepClip     = 1.0
)

// BregmanProximalGradient minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Bregman proksimalni
// gradijent: korak po svakoj dimenziji skaliran lokalnom procenom krivine |x_d|+0.1 (Bregman generator umesto
// euklidske norme), za razliku od proximal_gradient.go koji koristi istu skalu za sve dimenzije
// problem.Point je početna tačka pretrage
func BregmanProximalGradient(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := bregmanProximalGradientMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := bregmanProximalGradientLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := bregmanProximalGradientLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := bregmanProximalGradientTolerance
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

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		for d := 0; d < dimensions; d++ {
			scaleD := learningRate / (math.Abs(x[d]) + 0.1)
			candidate := proxL1Scalar(x[d]-scaleD*grad[d], scaleD*lambda)
			x[d] += clamp(candidate-x[d], -bregmanProximalGradientStepClip, bregmanProximalGradientStepClip)
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
		Method:     "bregman_proximal_gradient",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
