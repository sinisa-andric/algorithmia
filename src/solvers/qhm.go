package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	qhmLearningRate = 0.1
	qhmNu           = 0.7
	qhmMomentumCoef = 0.9
	qhmMaxSteps     = 1000
	qhmTolerance    = 1e-6
	qhmStepClip     = 1.0
)

// Qhm minimizuje konfigurisanu benchmark funkciju koristeći Quasi-Hyperbolic Momentum: korak je kvazi-hiperbolična
// mešavina trenutnog gradijenta i EMA momenta (koeficijent nu), bez ikakvog adaptivnog drugog momenta — razlika od
// qhadam.go koji ima i drugi moment, i od momentum.go koji koristi čist momentum bez mešanja sa gradijentom
// problem.Point je početna tačka pretrage
func Qhm(problem models.Problem) (result models.Result, err error) {

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

	learningRate := qhmLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	nu := qhmNu
	if v, ok := problem.Payload["nu"].(float64); ok {
		nu = v
	}

	momentumCoef := qhmMomentumCoef
	if v, ok := problem.Payload["momentum_coef"].(float64); ok {
		momentumCoef = v
	}

	maxSteps := qhmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := qhmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)

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
			m[d] = momentumCoef*m[d] + (1-momentumCoef)*grad[d]
			delta[d] = -learningRate * ((1-nu)*grad[d] + nu*m[d])
		}
		delta = clipStep(delta, qhmStepClip)
		for d := range x {
			x[d] += delta[d]
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
		Method:     "qhm",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
