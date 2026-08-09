package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	dykstraAlgorithmMaxSteps     = 1000
	dykstraAlgorithmLearningRate = 0.1
	dykstraAlgorithmLambda       = 0.01
	dykstraAlgorithmTolerance    = 1e-6
	dykstraAlgorithmRangeLow     = -5.0
	dykstraAlgorithmRangeHigh    = 5.0
)

// DykstraAlgorithm minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Dykstra algoritam: naizmenična
// projekcija na L1-prox i na kutiju [-5,5]^n uz akumulirane korekcione članove, za razliku od svih ostalih solvera
// koji ne rade eksplicitnu projekciju na granice tokom same iteracije, samo na kraju
// problem.Point je početna tačka pretrage
func DykstraAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := dykstraAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := dykstraAlgorithmLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := dykstraAlgorithmLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := dykstraAlgorithmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	p := make([]float64, dimensions)
	q := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)

		fStep := make([]float64, dimensions)
		for d := range fStep {
			fStep[d] = x[d] - learningRate*grad[d] + p[d]
		}
		y := proxL1(fStep, learningRate*lambda)

		pNew := make([]float64, dimensions)
		for d := range pNew {
			pNew[d] = fStep[d] - y[d]
		}

		xNew := make([]float64, dimensions)
		for d := range xNew {
			xNew[d] = clamp(y[d]+q[d], dykstraAlgorithmRangeLow, dykstraAlgorithmRangeHigh)
		}

		qNew := make([]float64, dimensions)
		for d := range qNew {
			qNew[d] = y[d] + q[d] - xNew[d]
		}

		x = xNew
		p = pNew
		q = qNew

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
		Method:     "dykstra_algorithm",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
