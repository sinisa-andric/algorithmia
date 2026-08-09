package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	trustRegionSubspaceMaxSteps  = 1000
	trustRegionSubspaceTolerance = 1e-6
	trustRegionSubspaceRadius    = 1.0
	trustRegionSubspaceMaxRadius = 10.0
	trustRegionSubspaceMinRadius = 1e-8
)

// trustRegionSubspaceGrid su (a,b) koeficijenti probani u 2D potprostoru razapetom sa {grad, prethodni_p}
var trustRegionSubspaceGrid = []float64{-1.0, -0.5, 0.0, 0.5, 1.0}

// trustRegionSubspaceNormalize vraća jedinični vektor u pravcu v, ili nula-vektor ako je v (skoro) nula
func trustRegionSubspaceNormalize(v []float64) []float64 {

	n := norm(v)
	result := make([]float64, len(v))
	if n < 1e-10 {
		return result
	}
	for d := range result {
		result[d] = v[d] / n
	}

	return result
}

// TrustRegionSubspace minimizuje konfigurisanu benchmark funkciju koristeći trust-region metod koji minimizuje
// model U CELOM 2D POTPROSTORU razapetom sa {gradijent, prethodni korak} (grid pretraga koeficijenata a,b), za
// razliku od trust_region_dogleg.go koji bira korak DUŽ JEDNE linije (Cauchy-Newton putanja)
// problem.Point je početna tačka pretrage
func TrustRegionSubspace(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := trustRegionSubspaceMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := trustRegionSubspaceTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	radius := trustRegionSubspaceRadius
	if v, ok := problem.Payload["radius"].(float64); ok {
		radius = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	x := append([]float64(nil), problem.Point...)
	prevP := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		gradNorm := trustRegionSubspaceNormalize(grad)
		prevPNorm := trustRegionSubspaceNormalize(prevP)

		fx := fn.Evaluate(x)
		bestTrial := make([]float64, dimensions)
		bestTrialValue := fx
		found := false
		for _, a := range trustRegionSubspaceGrid {
			for _, b := range trustRegionSubspaceGrid {
				candidate := make([]float64, dimensions)
				for d := range candidate {
					candidate[d] = a*radius*gradNorm[d] + b*radius*prevPNorm[d]
				}
				if cn := norm(candidate); cn > radius {
					for d := range candidate {
						candidate[d] = candidate[d] * radius / cn
					}
				}
				trial := make([]float64, dimensions)
				for d := range trial {
					trial[d] = x[d] + candidate[d]
				}
				trialValue := fn.Evaluate(trial)
				if trialValue < bestTrialValue {
					bestTrialValue = trialValue
					bestTrial = candidate
					found = true
				}
			}
		}

		p := bestTrial
		modelReduction := -dot(grad, p)

		rho := 0.0
		if found {
			rho = (fx - bestTrialValue) / (modelReduction + 1e-10)
		}

		if rho > 0.75 {
			radius = math.Min(radius*2, trustRegionSubspaceMaxRadius)
		}
		if rho < 0.25 {
			radius = math.Max(radius*0.5, trustRegionSubspaceMinRadius)
		}
		if rho > 0 {
			for d := range x {
				x[d] += p[d]
			}
			prevP = p
		}

		value := fn.Evaluate(x)
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
		Method:     "trust_region_subspace",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
