package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	palmMaxSteps     = 1000
	palmLearningRate = 0.1
	palmLambda       = 0.01
	palmTolerance    = 1e-6
	palmStepClip     = 1.0
)

// Palm minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći PALM: naizmenična proksimalna
// linearizovana minimizacija po blokovima dimenzija — prva polovina dimenzija se ažurira dok je druga fiksirana, pa
// obrnuto, sa gradijentom ponovo izračunatim između blokova
// problem.Point je početna tačka pretrage
func Palm(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := palmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := palmLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := palmLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := palmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	half := dimensions / 2
	block1 := make([]int, 0, half)
	block2 := make([]int, 0, dimensions-half)
	for d := 0; d < dimensions; d++ {
		if d < half {
			block1 = append(block1, d)
		} else {
			block2 = append(block2, d)
		}
	}

	x := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		for _, d := range block1 {
			candidate := proxL1Scalar(x[d]-learningRate*grad[d], learningRate*lambda)
			x[d] += clamp(candidate-x[d], -palmStepClip, palmStepClip)
		}

		grad = numGrad(fn.Evaluate, x, proximalGradEps)
		for _, d := range block2 {
			candidate := proxL1Scalar(x[d]-learningRate*grad[d], learningRate*lambda)
			x[d] += clamp(candidate-x[d], -palmStepClip, palmStepClip)
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
		Method:     "palm",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
