package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	proximalAdmmMaxSteps     = 1000
	proximalAdmmLearningRate = 0.1
	proximalAdmmLambda       = 0.01
	proximalAdmmRho          = 1.0
	proximalAdmmTauProx      = 0.1
	proximalAdmmTolerance    = 1e-6
	proximalAdmmStepClip     = 1.0
)

// ProximalAdmm minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći proksimalni ADMM: x-podproblem
// ima dodatni kvadratni prigušujući član (tau_prox/2)*||x-x_prev||^2 usidren na vrednost x iz PRETHODNE iteracije,
// za razliku od admm.go koji nema ovaj dodatni stabilizacioni član
// problem.Point je početna tačka pretrage
func ProximalAdmm(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := proximalAdmmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := proximalAdmmLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := proximalAdmmLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	rho := proximalAdmmRho
	if v, ok := problem.Payload["rho"].(float64); ok {
		rho = v
	}

	tauProx := proximalAdmmTauProx
	if v, ok := problem.Payload["tau_prox"].(float64); ok {
		tauProx = v
	}

	tolerance := proximalAdmmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	xPrev := append([]float64(nil), problem.Point...)
	z := append([]float64(nil), problem.Point...)
	u := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		// x_prev mora čuvati vrednost x iz PRETHODNE iteracije (ne trenutnu) da bi tau_prox član stvarno
		// prigušivao pomeraj u odnosu na prošli korak — ažuriranje x_prev se zato dešava PRE promene x, a ne
		// neposredno pre upotrebe u istoj formuli (što bi dalo (x-x_prev)=0 svaki put i poništilo član)
		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = -learningRate * (grad[d] + rho*(x[d]-z[d]+u[d]) + tauProx*(x[d]-xPrev[d]))
		}
		delta = clipStep(delta, proximalAdmmStepClip)

		xPrev = append([]float64(nil), x...)
		for d := range x {
			x[d] += delta[d]
		}

		zRaw := make([]float64, dimensions)
		for d := range zRaw {
			zRaw[d] = x[d] + u[d]
		}
		z = proxL1(zRaw, lambda/(rho+1e-10))

		for d := range u {
			u[d] += x[d] - z[d]
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
		Method:     "proximal_admm",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
