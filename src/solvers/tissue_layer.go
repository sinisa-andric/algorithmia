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
	tissueLayerPopulation = 20
	tissueLayerMaxSteps   = 500
	tissueLayerTolerance  = 1e-6
	tissueLayerRangeLow   = -5.0
	tissueLayerRangeHigh  = 5.0
)

// TissueLayer minimizuje konfigurisanu benchmark funkciju koristeći Tissue-Layer Immune Algorithm: ćelije se
// svakog koraka rangiraju i dele u tri sloja — unutrašnji fino pretražuje sopstvenu okolinu, srednji se kreće ka
// najboljem rešenju i nasumičnom peer-u, a spoljni sloj slobodno istražuje širokim nasumičnim koracima
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func TissueLayer(problem models.Problem) (result models.Result, err error) {

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

	population := tissueLayerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := tissueLayerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := tissueLayerTolerance
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
			individual[d] = tissueLayerRangeLow + rand.Float64()*(tissueLayerRangeHigh-tissueLayerRangeLow)
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

		third := population / 3
		innerEnd := third
		middleEnd := 2 * third

		for r := 0; r < population; r++ {
			idx := order[r]

			switch {
			case r < innerEnd:
				for d := range individuals[idx] {
					amplitude := 0.15 * (1 - float64(steps)/float64(maxSteps))
					individuals[idx][d] = clamp(individuals[idx][d]+(rand.Float64()*2-1)*amplitude, tissueLayerRangeLow, tissueLayerRangeHigh)
				}
			case r < middleEnd:
				peerIdx := idx
				if population > 1 {
					peerIdx = rand.IntN(population)
					for peerIdx == idx {
						peerIdx = rand.IntN(population)
					}
				}
				for d := range individuals[idx] {
					pullBest := rand.Float64() * (best[d] - individuals[idx][d])
					pullPeer := rand.Float64() * (individuals[peerIdx][d] - individuals[idx][d]) * 0.3
					individuals[idx][d] = clamp(individuals[idx][d]+pullBest+pullPeer, tissueLayerRangeLow, tissueLayerRangeHigh)
				}
			default:
				for d := range individuals[idx] {
					individuals[idx][d] = clamp(individuals[idx][d]+(rand.Float64()*2-1)*1.2, tissueLayerRangeLow, tissueLayerRangeHigh)
				}
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
		Method:     "tissue_layer",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
