package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	optimalGradientMethodMaxSteps     = 1000
	optimalGradientMethodLearningRate = 0.1
	optimalGradientMethodLambda       = 0.01
	optimalGradientMethodTolerance    = 1e-6
	optimalGradientMethodStepClip     = 1.0
)

// OptimalGradientMethod minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Optimal Gradient Method
// (Kim-Fessler): kao fista.go, ali poslednji korak dobija duplo veći momentum koeficijent (faktor 8 umesto 4 u
// rekurziji za theta), dajući dodatni "bonus" ubrzanje na samom kraju izvršavanja
// problem.Point je početna tačka pretrage
func OptimalGradientMethod(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := optimalGradientMethodMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := optimalGradientMethodLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := optimalGradientMethodLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := optimalGradientMethodTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	y := append([]float64(nil), x...)
	theta := 1.0

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, y, proximalGradEps)
		raw := make([]float64, dimensions)
		for d := range raw {
			raw[d] = y[d] - learningRate*grad[d]
		}
		xProposed := proxL1(raw, learningRate*lambda)

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xProposed[d] - x[d]
		}
		delta = clipStep(delta, optimalGradientMethodStepClip)

		xPrev := append([]float64(nil), x...)
		for d := range x {
			x[d] += delta[d]
		}

		var thetaNew float64
		if steps == maxSteps-1 {
			thetaNew = (1 + math.Sqrt(1+8*theta*theta)) / 2
		} else {
			thetaNew = (1 + math.Sqrt(1+4*theta*theta)) / 2
		}
		for d := range y {
			y[d] = x[d] + ((theta-1)/thetaNew)*(x[d]-xPrev[d])
		}
		theta = thetaNew

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
		Method:     "optimal_gradient_method",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
