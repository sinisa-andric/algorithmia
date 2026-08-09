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
	immuneNetworkPopulation           = 20
	immuneNetworkMaxSteps             = 500
	immuneNetworkSuppressionThreshold = 0.5
	immuneNetworkCloneFactor          = 5
	immuneNetworkTolerance            = 1e-6
	immuneNetworkRangeLow             = -5.0
	immuneNetworkRangeHigh            = 5.0
)

// ImmuneNetwork minimizuje konfigurisanu benchmark funkciju koristeći Optimized Artificial Immune Network: svako
// antitelo se klonira i hipermutira jačinom obrnuto proporcionalnom rangu afiniteta kao u klonalnoj selekciji, a
// zatim se antitela bliža jedno drugom od praga supresije međusobno potiskuju pre dopune populacije novim jedinkama
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func ImmuneNetwork(problem models.Problem) (result models.Result, err error) {

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

	population := immuneNetworkPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := immuneNetworkMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	suppressionThreshold := immuneNetworkSuppressionThreshold
	if v, ok := problem.Payload["suppression_threshold"].(float64); ok {
		suppressionThreshold = v
	}

	cloneFactor := immuneNetworkCloneFactor
	if v, ok := problem.Payload["clone_factor"].(float64); ok {
		cloneFactor = int(v)
	}

	tolerance := immuneNetworkTolerance
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
			individual[d] = immuneNetworkRangeLow + rand.Float64()*(immuneNetworkRangeHigh-immuneNetworkRangeLow)
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

		for rank, idx := range order {
			mutationScale := 0.05
			if len(individuals) > 1 {
				mutationScale = 0.05 + (float64(rank)/float64(len(individuals)-1))*0.95
			}

			bestClone := individuals[idx]
			bestCloneValue := values[idx]
			for range cloneFactor {
				clone := make([]float64, dimensions)
				for d := range clone {
					clone[d] = clamp(individuals[idx][d]+rand.NormFloat64()*mutationScale, immuneNetworkRangeLow, immuneNetworkRangeHigh)
				}
				cloneValue := fn.Evaluate(clone)
				if cloneValue < bestCloneValue {
					bestCloneValue = cloneValue
					bestClone = clone
				}
			}

			if bestCloneValue < values[idx] {
				individuals[idx] = bestClone
				values[idx] = bestCloneValue
			}
		}

		// mrežna supresija: antitela bliža jedno drugom od praga se međusobno potiskuju, lošije od para se uklanja
		removed := make([]bool, len(individuals))
		for i := 0; i < len(individuals); i++ {
			if removed[i] {
				continue
			}
			for j := i + 1; j < len(individuals); j++ {
				if removed[j] {
					continue
				}
				diff := make([]float64, dimensions)
				for d := range diff {
					diff[d] = individuals[i][d] - individuals[j][d]
				}
				if norm(diff) < suppressionThreshold {
					if values[i] <= values[j] {
						removed[j] = true
					} else {
						removed[i] = true
						break
					}
				}
			}
		}

		survivors := make([][]float64, 0, len(individuals))
		survivorValues := make([]float64, 0, len(individuals))
		for i := range individuals {
			if !removed[i] {
				survivors = append(survivors, individuals[i])
				survivorValues = append(survivorValues, values[i])
			}
		}

		// dopuna populacije nasumičnim novim antitelima do prvobitne veličine radi diverziteta
		for len(survivors) < population {
			individual := make([]float64, dimensions)
			for d := range individual {
				individual[d] = immuneNetworkRangeLow + rand.Float64()*(immuneNetworkRangeHigh-immuneNetworkRangeLow)
			}
			survivors = append(survivors, individual)
			survivorValues = append(survivorValues, fn.Evaluate(individual))
		}

		individuals = survivors
		values = survivorValues

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
		Method:     "immune_network",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
