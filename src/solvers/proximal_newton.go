package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	proximalNewtonMaxSteps     = 1000
	proximalNewtonLearningRate = 0.1
	proximalNewtonLambda       = 0.01
	proximalNewtonTolerance    = 1e-6
	proximalNewtonStepClip     = 1.0
	proximalNewtonHessEps      = 1e-3
)

// ProximalNewton minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći proksimalni Newton metod:
// dijagonalna aproksimacija Hesijana (numerička druga izvod po svakoj dimenziji) daje per-koordinatni adaptivni
// korak, za razliku od svih ostalih proksimalnih solvera koji koriste isti learning_rate za sve dimenzije
// problem.Point je početna tačka pretrage
func ProximalNewton(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := proximalNewtonMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := proximalNewtonLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := proximalNewtonLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := proximalNewtonTolerance
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
		fx := fn.Evaluate(x)

		for d := 0; d < dimensions; d++ {
			plus := append([]float64(nil), x...)
			minus := append([]float64(nil), x...)
			plus[d] += proximalNewtonHessEps
			minus[d] -= proximalNewtonHessEps
			hessD := (fn.Evaluate(plus) - 2*fx + fn.Evaluate(minus)) / (proximalNewtonHessEps * proximalNewtonHessEps)

			stepD := learningRate / (math.Abs(hessD) + 1.0)
			candidateD := proxL1Scalar(x[d]-stepD*grad[d], stepD*lambda)
			x[d] += clamp(candidateD-x[d], -proximalNewtonStepClip, proximalNewtonStepClip)
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
		Method:     "proximal_newton",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
