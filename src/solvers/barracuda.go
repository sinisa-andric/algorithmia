package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	barracudaPopulation = 20
	barracudaMaxSteps   = 1000
	barracudaTolerance  = 1e-6
	barracudaRangeLow   = -5.0
	barracudaRangeHigh  = 5.0
)

// Barracuda minimizuje konfigurisanu benchmark funkciju koristeći Barracuda Optimization Algorithm: barakuda ili
// prati jato krećući se ka centru mase i ka najboljem rešenju, ili izvodi brz i snažan napad direktno ka najboljem
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Barracuda(problem models.Problem) (result models.Result, err error) {

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

	population := barracudaPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := barracudaMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := barracudaTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	barracudas := make([][]float64, population)
	values := make([]float64, population)
	for i := range barracudas {
		barracuda := make([]float64, dimensions)
		for d := range barracuda {
			barracuda[d] = barracudaRangeLow + rand.Float64()*(barracudaRangeHigh-barracudaRangeLow)
		}
		barracudas[i] = barracuda
		values[i] = fn.Evaluate(barracuda)
	}

	best := append([]float64(nil), barracudas[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), barracudas[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		center := make([]float64, dimensions)
		for _, b := range barracudas {
			for d := range center {
				center[d] += b[d]
			}
		}
		for d := range center {
			center[d] /= float64(population)
		}

		for i := range barracudas {
			if rand.Float64() < 0.5 {
				for d := range barracudas[i] {
					barracudas[i][d] = barracudas[i][d] + rand.Float64()*(center[d]-barracudas[i][d]) + rand.Float64()*(best[d]-barracudas[i][d])*0.5
				}
			} else {
				for d := range barracudas[i] {
					barracudas[i][d] = barracudas[i][d] + 0.9*(best[d]-barracudas[i][d])
				}
			}

			// Obe grane su čisto konvergentne (bez ikakvog šuma) — jato se brzo konsoliduje oko best-a
			// pre nego što stigne blizu optimuma, a bez preostalog diverziteta dalje napredovanje staje.
			// Povremena opadajuća perturbacija održava istraživanje dovoljno dugo da se to izbegne
			if rand.Float64() < 0.15 {
				for d := range barracudas[i] {
					barracudas[i][d] = barracudas[i][d] + (rand.Float64()*2-1)*0.5*(1-float64(steps)/float64(maxSteps))
				}
			}

			for d := range barracudas[i] {
				barracudas[i][d] = clamp(barracudas[i][d], barracudaRangeLow, barracudaRangeHigh)
			}
			values[i] = fn.Evaluate(barracudas[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), barracudas[i]...)
			}
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
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "barracuda",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
