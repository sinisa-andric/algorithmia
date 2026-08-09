package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	waspPopulation = 20
	waspMaxSteps   = 1000
	waspTolerance  = 1e-6
	waspRangeLow   = -5.0
	waspRangeHigh  = 5.0
)

// Wasp minimizuje konfigurisanu benchmark funkciju koristeći Wasp Swarm Optimization: populacija se sortira u
// hijerarhiju dominacije, svaka osa se kreće ka osi neposredno boljeg ranga (najbolja osa ka sebi),
// sa uticajem koji opada što je rang bliži najboljem
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Wasp(problem models.Problem) (result models.Result, err error) {

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

	population := waspPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := waspMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := waspTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sortByValue := func(points [][]float64, vals []float64) {
		idx := make([]int, len(points))
		for i := range idx {
			idx[i] = i
		}
		for i := 1; i < len(idx); i++ {
			for j := i; j > 0 && vals[idx[j]] < vals[idx[j-1]]; j-- {
				idx[j], idx[j-1] = idx[j-1], idx[j]
			}
		}
		sortedPoints := make([][]float64, len(points))
		sortedVals := make([]float64, len(points))
		for i, k := range idx {
			sortedPoints[i] = points[k]
			sortedVals[i] = vals[k]
		}
		copy(points, sortedPoints)
		copy(vals, sortedVals)
	}

	wasps := make([][]float64, population)
	values := make([]float64, population)
	for i := range wasps {
		wasp := make([]float64, dimensions)
		for d := range wasp {
			wasp[d] = waspRangeLow + rand.Float64()*(waspRangeHigh-waspRangeLow)
		}
		wasps[i] = wasp
		values[i] = fn.Evaluate(wasp)
	}
	sortByValue(wasps, values)

	best := append([]float64(nil), wasps[0]...)
	bestValue := values[0]

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range wasps {
			if i == 0 {
				// Osa najboljeg ranga nema boljeg dominanta (dominant=best=ona sama), pa bi bez
				// ove lokalne pretrage ostala zauvek zamrznuta na inicijalnoj poziciji — a pošto
				// se svi ostali kreću ka njoj isključivo konveksnom interpolacijom (nikad je ne
				// prestižu), ceo roj bi bio ograničen kvalitetom te prve nasumične tačke.
				for d := range wasps[i] {
					wasps[i][d] = wasps[i][d] + (rand.Float64()*2-1)*0.5*(1-float64(steps)/float64(maxSteps))
					wasps[i][d] = clamp(wasps[i][d], waspRangeLow, waspRangeHigh)
				}
			} else {
				dominant := wasps[i-1]
				for d := range wasps[i] {
					wasps[i][d] = wasps[i][d] + rand.Float64()*(dominant[d]-wasps[i][d])*(1-float64(i)/float64(population))
					wasps[i][d] = clamp(wasps[i][d], waspRangeLow, waspRangeHigh)
				}
			}
			values[i] = fn.Evaluate(wasps[i])
		}

		sortByValue(wasps, values)
		bestValue = values[0]
		best = append([]float64(nil), wasps[0]...)

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
		Method:     "wasp",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
