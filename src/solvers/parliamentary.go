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
	parliamentaryPopulation     = 20
	parliamentaryMaxSteps       = 500
	parliamentaryCoalitionRatio = 0.4
	parliamentaryTolerance      = 1e-6
	parliamentaryRangeLow       = -5.0
	parliamentaryRangeHigh      = 5.0
)

// Parliamentary minimizuje konfigurisanu benchmark funkciju koristeći Parliamentary Optimization Algorithm:
// najbolje rangirane stranke formiraju koaliciju i konvergiraju ka vodećoj (best-u), dok opozicija širom
// eksploriše prostor uz slabu vezu ka vodećoj stranci
// problem.Point inicijalizuje populaciju stranaka i određuje njenu dimenzionalnost
func Parliamentary(problem models.Problem) (result models.Result, err error) {

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

	population := parliamentaryPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := parliamentaryMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	coalitionRatio := parliamentaryCoalitionRatio
	if v, ok := problem.Payload["coalition_ratio"].(float64); ok {
		coalitionRatio = v
	}

	tolerance := parliamentaryTolerance
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
			individual[d] = parliamentaryRangeLow + rand.Float64()*(parliamentaryRangeHigh-parliamentaryRangeLow)
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

		order := make([]int, len(individuals))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		granica := int(math.Round(coalitionRatio * float64(len(individuals))))
		if granica < 0 {
			granica = 0
		}
		if granica > len(order) {
			granica = len(order)
		}

		for r, idx := range order {
			if r < granica {
				for d := range individuals[idx] {
					pull := rand.Float64() * (best[d] - individuals[idx][d]) * 0.5
					noise := (rand.Float64()*2 - 1) * 0.05
					individuals[idx][d] = clamp(individuals[idx][d]+pull+noise, parliamentaryRangeLow, parliamentaryRangeHigh)
				}
			} else {
				for d := range individuals[idx] {
					noise := (rand.Float64()*2 - 1) * 0.6
					pull := rand.Float64() * (best[d] - individuals[idx][d]) * 0.15
					individuals[idx][d] = clamp(individuals[idx][d]+noise+pull, parliamentaryRangeLow, parliamentaryRangeHigh)
				}
			}
			values[idx] = fn.Evaluate(individuals[idx])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "parliamentary",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
