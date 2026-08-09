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
	tCellPopulation      = 20
	tCellMaxSteps        = 500
	tCellRegulatoryRatio = 0.3
	tCellTolerance       = 1e-6
	tCellRangeLow        = -5.0
	tCellRangeHigh       = 5.0
)

// TCell minimizuje konfigurisanu benchmark funkciju koristeći Adaptive T-cell Regulatory Algorithm: ćelije se
// svakog koraka rangiraju po vrednosti, bolje rangirane efektorske ćelije direktno napadaju (kreću se ka)
// najboljem rešenju, dok preostale regulatorne ćelije održavaju diverzitet kretanjem ka nasumičnom peer-u uz šum
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func TCell(problem models.Problem) (result models.Result, err error) {

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

	population := tCellPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := tCellMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	regulatoryRatio := tCellRegulatoryRatio
	if v, ok := problem.Payload["regulatory_ratio"].(float64); ok {
		regulatoryRatio = v
	}

	tolerance := tCellTolerance
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
			individual[d] = tCellRangeLow + rand.Float64()*(tCellRangeHigh-tCellRangeLow)
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

		order := make([]int, population)
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		granica := int(math.Round(regulatoryRatio * float64(population)))
		if granica > population {
			granica = population
		}
		numEffector := population - granica

		for r := 0; r < numEffector; r++ {
			idx := order[r]
			for d := range individuals[idx] {
				individuals[idx][d] = clamp(individuals[idx][d]+0.5*(best[d]-individuals[idx][d]), tCellRangeLow, tCellRangeHigh)
			}
			values[idx] = fn.Evaluate(individuals[idx])
		}

		for r := numEffector; r < population; r++ {
			idx := order[r]
			peerIdx := idx
			if population > 1 {
				peerIdx = rand.IntN(population)
				for peerIdx == idx {
					peerIdx = rand.IntN(population)
				}
			}
			for d := range individuals[idx] {
				pull := rand.Float64() * (individuals[peerIdx][d] - individuals[idx][d]) * 0.2
				individuals[idx][d] = clamp(individuals[idx][d]+(rand.Float64()*2-1)*0.5+pull, tCellRangeLow, tCellRangeHigh)
			}
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
		Method:     "t_cell",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
