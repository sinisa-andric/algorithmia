package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	pdhgChambollePockMaxSteps  = 1000
	pdhgChambollePockTau       = 0.1
	pdhgChambollePockSigma     = 0.1
	pdhgChambollePockTheta     = 1.0
	pdhgChambollePockLambda    = 0.01
	pdhgChambollePockTolerance = 1e-6
	pdhgChambollePockStepClip  = 1.0
)

// PdhgChambollePock minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Primal-Dual Hybrid Gradient
// (Chambolle-Pock): umesto direktnog proksimalnog koraka na L1 članu, drži dualnu promenljivu y projektovanu na
// L-beskonačno loptu poluprečnika lambda, i ekstrapolira primarnu tačku faktorom theta
// problem.Point je početna tačka pretrage
func PdhgChambollePock(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := pdhgChambollePockMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tau := pdhgChambollePockTau
	if v, ok := problem.Payload["tau"].(float64); ok {
		tau = v
	}

	sigma := pdhgChambollePockSigma
	if v, ok := problem.Payload["sigma"].(float64); ok {
		sigma = v
	}

	theta := pdhgChambollePockTheta
	if v, ok := problem.Payload["theta"].(float64); ok {
		theta = v
	}

	lambda := pdhgChambollePockLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := pdhgChambollePockTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	xBar := append([]float64(nil), x...)
	y := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for d := range y {
			y[d] = clamp(y[d]+sigma*xBar[d], -lambda, lambda)
		}

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		xProposed := make([]float64, dimensions)
		for d := range xProposed {
			xProposed[d] = x[d] - tau*(grad[d]+y[d])
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xProposed[d] - x[d]
		}
		delta = clipStep(delta, pdhgChambollePockStepClip)

		xPrev := append([]float64(nil), x...)
		for d := range x {
			x[d] += delta[d]
		}
		for d := range xBar {
			xBar[d] = x[d] + theta*(x[d]-xPrev[d])
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
		Method:     "pdhg_chambolle_pock",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
