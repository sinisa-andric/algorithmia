package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	proximalPointMaxSteps     = 1000
	proximalPointLearningRate = 0.1
	proximalPointLambda       = 0.01
	proximalPointInnerSteps   = 5
	proximalPointTolerance    = 1e-6
	proximalPointStepClip     = 1.0
)

// ProximalPoint minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Proximal Point Algorithm: svaki
// spoljni korak rešava kompletan potproblem sa inner_steps unutrašnjih proksimalno-gradijentnih iteracija
// usidrenih na tačku sa početka te spoljne iteracije, za razliku od proximal_gradient.go koji radi tačno jedan
// gradijent+prox korak po iteraciji
// problem.Point je početna tačka pretrage
func ProximalPoint(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := proximalPointMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := proximalPointLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := proximalPointLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	innerSteps := proximalPointInnerSteps
	if v, ok := problem.Payload["inner_steps"].(float64); ok {
		innerSteps = int(v)
	}

	tolerance := proximalPointTolerance
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

		anchor := append([]float64(nil), x...)
		for inner := 0; inner < innerSteps; inner++ {
			grad := numGrad(fn.Evaluate, x, proximalGradEps)
			raw := make([]float64, dimensions)
			for d := range raw {
				raw[d] = x[d] - learningRate*(grad[d]+(x[d]-anchor[d]))
			}
			candidate := proxL1(raw, learningRate*lambda)

			delta := make([]float64, dimensions)
			for d := range delta {
				delta[d] = candidate[d] - x[d]
			}
			delta = clipStep(delta, proximalPointStepClip)
			for d := range x {
				x[d] += delta[d]
			}
		}

		value := objective(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			// bestValue prati objective (fn + L1 penal), dok rezultat vraća čist fn.Evaluate(best) — koristi se
			// isti izraz kao Value: ispod da bi putanja bila konzistentna sa finalnim rezultatom
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
		Method:     "proximal_point",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
