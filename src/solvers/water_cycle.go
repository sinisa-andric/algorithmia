package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

const (
	waterCyclePopulation = 20
	waterCycleMaxSteps   = 500
	waterCycleNumRivers  = 4
	waterCycleTolerance  = 1e-6
	waterCycleRangeLow   = -5.0
	waterCycleRangeHigh  = 5.0
	waterCycleFlowSpeed  = 2.0
)

// WaterCycle minimizuje konfigurisanu benchmark funkciju koristeći Water Cycle Algorithm: potoci teku ka svojoj
// dodeljenoj reci a reke ka moru (najboljem rešenju), dok se jedinke koje se previše približe moru isparavaju i
// padaju kao nova kiša nasumično u prostoru
// problem.Point inicijalizuje populaciju kapi i određuje njenu dimenzionalnost
func WaterCycle(problem models.Problem) (result models.Result, err error) {

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

	population := waterCyclePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := waterCycleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	numRivers := waterCycleNumRivers
	if v, ok := problem.Payload["num_rivers"].(float64); ok {
		numRivers = int(v)
	}

	tolerance := waterCycleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = waterCycleRangeLow + rand.Float64()*(waterCycleRangeHigh-waterCycleRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		if numRivers < 1 {
			numRivers = 1
		}
		if numRivers > len(individuals)-1 {
			numRivers = len(individuals) - 1
		}

		order := make([]int, len(individuals))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		seaIdx := order[0]
		riverIdxs := order[1 : 1+numRivers]
		streamIdxs := order[1+numRivers:]

		threshold := 0.1 * (1 - float64(steps)/float64(maxSteps))

		// more (najbolje rangirana jedinka) i dalje mora imati putanju ažuriranja — dobija finu lokalnu
		// pretragu umesto da ostane zamrznuto na inicijalnoj poziciji
		for d := range individuals[seaIdx] {
			individuals[seaIdx][d] = clamp(individuals[seaIdx][d]+(rand.Float64()*2-1)*0.05, waterCycleRangeLow, waterCycleRangeHigh)
		}
		values[seaIdx] = fn.Evaluate(individuals[seaIdx])

		for _, idx := range riverIdxs {
			for d := range individuals[idx] {
				pull := rand.Float64() * waterCycleFlowSpeed * (individuals[seaIdx][d] - individuals[idx][d])
				noise := (rand.Float64()*2 - 1) * 0.05
				individuals[idx][d] = clamp(individuals[idx][d]+pull+noise, waterCycleRangeLow, waterCycleRangeHigh)
			}
			values[idx] = fn.Evaluate(individuals[idx])
		}

		for k, idx := range streamIdxs {
			riverIdx := riverIdxs[k%len(riverIdxs)]
			for d := range individuals[idx] {
				pull := rand.Float64() * waterCycleFlowSpeed * (individuals[riverIdx][d] - individuals[idx][d])
				noise := (rand.Float64()*2 - 1) * 0.05
				individuals[idx][d] = clamp(individuals[idx][d]+pull+noise, waterCycleRangeLow, waterCycleRangeHigh)
			}
			values[idx] = fn.Evaluate(individuals[idx])
		}

		for _, idx := range append(append([]int{}, riverIdxs...), streamIdxs...) {
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = individuals[idx][d] - individuals[seaIdx][d]
			}
			if norm(diff) < threshold {
				individual := make([]float64, dimensions)
				for d := range individual {
					individual[d] = waterCycleRangeLow + rand.Float64()*(waterCycleRangeHigh-waterCycleRangeLow)
				}
				individuals[idx] = individual
				values[idx] = fn.Evaluate(individual)
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
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
		Method:     "water_cycle",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
