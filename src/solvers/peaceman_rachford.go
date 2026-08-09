package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	peacemanRachfordMaxSteps     = 1000
	peacemanRachfordLearningRate = 0.1
	peacemanRachfordLambda       = 0.01
	peacemanRachfordTolerance    = 1e-6
	peacemanRachfordStepClip     = 1.0
)

// PeacemanRachford minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Peaceman-Rachford splitting:
// puna (neprigušena) naizmenična refleksija oko glatkog člana f i L1 člana g, agresivnija od Douglas-Rachford
// splitting-a koji ima prigušujući faktor 0.5 u finalnoj korekciji
// problem.Point je početna tačka pretrage
func PeacemanRachford(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := peacemanRachfordMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := peacemanRachfordLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := peacemanRachfordLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := peacemanRachfordTolerance
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
		r1 := make([]float64, dimensions)
		for d := range r1 {
			r1[d] = 2*(x[d]-learningRate*grad[d]) - x[d]
		}
		proxed := proxL1(r1, learningRate*lambda)
		r2 := make([]float64, dimensions)
		for d := range r2 {
			r2[d] = 2*proxed[d] - r1[d]
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = r2[d] - x[d]
		}
		// ova varijanta je sklona oscilaciji bez ograničenja jer puna (neprigušena) refleksija može
		// naizmenično prebacivati preko optimuma sa rastućom amplitudom na loše uslovljenim funkcijama
		delta = clipStep(delta, peacemanRachfordStepClip)
		for d := range x {
			x[d] += delta[d]
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
		Method:     "peaceman_rachford",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
