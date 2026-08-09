package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	proximalGradientBbMaxSteps    = 1000
	proximalGradientBbLambda      = 0.01
	proximalGradientBbTolerance   = 1e-6
	proximalGradientBbStepClip    = 1.0
	proximalGradientBbInitialStep = 0.1
	proximalGradientBbStepMin     = 0.001
	proximalGradientBbStepMax     = 1.0
)

// ProximalGradientBb minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći proksimalni gradijent sa
// Barzilai-Borwein korakom: korak se računa iz promene pozicije i gradijenta poslednja dva koraka, bez fiksnog
// learning_rate parametra koji imaju svi ostali solveri u ovom modulu
// problem.Point je početna tačka pretrage
func ProximalGradientBb(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := proximalGradientBbMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	lambda := proximalGradientBbLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := proximalGradientBbTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	xPrev := append([]float64(nil), problem.Point...)
	gradPrev := make([]float64, dimensions)
	step := proximalGradientBbInitialStep

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)

		if steps > 0 {
			sDotS, sDotY := 0.0, 0.0
			for d := range x {
				s := x[d] - xPrev[d]
				y := grad[d] - gradPrev[d]
				sDotS += s * s
				sDotY += s * y
			}
			step = sDotS / (sDotY + 1e-10)
			step = clamp(step, proximalGradientBbStepMin, proximalGradientBbStepMax)
		}

		raw := make([]float64, dimensions)
		for d := range raw {
			raw[d] = x[d] - step*grad[d]
		}
		xNew := proxL1(raw, step*lambda)

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xNew[d] - x[d]
		}
		delta = clipStep(delta, proximalGradientBbStepClip)

		xPrev = append([]float64(nil), x...)
		gradPrev = grad
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
		Method:     "proximal_gradient_bb",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
