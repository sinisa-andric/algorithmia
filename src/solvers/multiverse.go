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
	multiversePopulation = 20
	multiverseMaxSteps   = 500
	multiverseWepMin     = 0.2
	multiverseWepMax     = 1.0
	multiverseTolerance  = 1e-6
	multiverseRangeLow   = -5.0
	multiverseRangeHigh  = 5.0
)

// Multiverse minimizuje konfigurisanu benchmark funkciju koristeći Multi-Verse Optimizer: univerzumi sa višom
// normalizovanom inflacionom stopom (lošiji od best-a) uvoze dimenzije iz boljih univerzuma preko crne/bele rupe,
// a crvotočina dodatno prenosi svaki univerzum direktno u okolinu best-a sa verovatnoćom i dometom koji se menjaju
// tokom izvršavanja
// problem.Point inicijalizuje populaciju univerzuma i određuje njenu dimenzionalnost
func Multiverse(problem models.Problem) (result models.Result, err error) {

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

	population := multiversePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := multiverseMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	wepMin := multiverseWepMin
	if v, ok := problem.Payload["wep_min"].(float64); ok {
		wepMin = v
	}

	wepMax := multiverseWepMax
	if v, ok := problem.Payload["wep_max"].(float64); ok {
		wepMax = v
	}

	tolerance := multiverseTolerance
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
			individual[d] = multiverseRangeLow + rand.Float64()*(multiverseRangeHigh-multiverseRangeLow)
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

	wormhole := func(idx int, wep, tdr float64) {
		for d := range individuals[idx] {
			if rand.Float64() < wep {
				r3 := rand.Float64()
				r4 := rand.Float64()
				span := (multiverseRangeHigh-multiverseRangeLow)*r4 + multiverseRangeLow
				if r3 < 0.5 {
					individuals[idx][d] = best[d] + tdr*span
				} else {
					individuals[idx][d] = best[d] - tdr*span
				}
			}
			individuals[idx][d] = clamp(individuals[idx][d], multiverseRangeLow, multiverseRangeHigh)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		wep := wepMin + float64(steps)/float64(maxSteps)*(wepMax-wepMin)
		tdr := 1 - math.Pow(float64(steps), 1.0/6.0)/math.Pow(float64(maxSteps), 1.0/6.0)

		order := make([]int, len(individuals))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		worstValue := values[order[len(order)-1]]
		bestValueThisStep := values[order[0]]

		for rank, idx := range order {
			if rank == 0 {
				// najbolje rangiran univerzum nema od kog boljeg da uvozi dimenzije, ali i dalje mora imati
				// putanju ažuriranja — dobija samo crvotočinu ka best-u kao i ostali
				wormhole(idx, wep, tdr)
				values[idx] = fn.Evaluate(individuals[idx])
				continue
			}

			normalizedInflation := (values[idx] - bestValueThisStep) / (worstValue - bestValueThisStep + 1e-10)
			for d := range individuals[idx] {
				if rand.Float64() < normalizedInflation {
					sourceIdx := order[weightedIndex(len(order))]
					individuals[idx][d] = individuals[sourceIdx][d]
				}
			}
			wormhole(idx, wep, tdr)
			values[idx] = fn.Evaluate(individuals[idx])
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
		Method:     "multiverse",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
