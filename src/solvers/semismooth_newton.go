package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	semismoothNewtonMaxSteps     = 1000
	semismoothNewtonLearningRate = 0.1
	semismoothNewtonLambda       = 0.01
	semismoothNewtonTolerance    = 1e-6
	semismoothNewtonStepClip     = 1.0
)

// SemismoothNewton minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći poluglatki Newton metod na
// fiksnoj tački proksimalnog gradijenta: dodatni Newton-like korak se primenjuje samo na koordinate koje
// soft-thresholding nije ugasio na nulu (aktivni skup), dok se ugašene koordinate agresivnije istražuju, za
// razliku od svih ostalih solvera koji tretiraju sve dimenzije uniformno
// problem.Point je početna tačka pretrage
func SemismoothNewton(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := semismoothNewtonMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := semismoothNewtonLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := semismoothNewtonLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := semismoothNewtonTolerance
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
		raw := make([]float64, dimensions)
		for d := range raw {
			raw[d] = x[d] - learningRate*grad[d]
		}
		candidate := proxL1(raw, learningRate*lambda)

		for d := range x {
			var xNewD float64
			if candidate[d] != 0 {
				xNewD = candidate[d] - 0.5*learningRate*grad[d]
			} else {
				xNewD = candidate[d] + (rand.Float64()*2-1)*0.05
			}
			x[d] += clamp(xNewD-x[d], -semismoothNewtonStepClip, semismoothNewtonStepClip)
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
		Method:     "semismooth_newton",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
