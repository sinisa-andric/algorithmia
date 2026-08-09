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
	lightningSearchPopulation = 20
	lightningSearchMaxSteps   = 500
	lightningSearchTolerance  = 1e-6
	lightningSearchRangeLow   = -5.0
	lightningSearchRangeHigh  = 5.0
)

// LightningSearch minimizuje konfigurisanu benchmark funkciju koristeći Lightning Search Algorithm: najgorih 10%
// projektila izvodi probojni udar direktno u gausovsku okolinu best-a, dok ostali granaju korakom nasumične
// veličine usmerenim ka best-u, sa disperzijom koja opada tokom izvršavanja
// problem.Point inicijalizuje populaciju projektila i određuje njenu dimenzionalnost
func LightningSearch(problem models.Problem) (result models.Result, err error) {

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

	population := lightningSearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := lightningSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := lightningSearchTolerance
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
			individual[d] = lightningSearchRangeLow + rand.Float64()*(lightningSearchRangeHigh-lightningSearchRangeLow)
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

		energy := 2.05 - 2*float64(steps)/float64(maxSteps)

		numStrike := population / 10
		strikeSet := make(map[int]bool, numStrike)
		if numStrike > 0 {
			order := make([]int, population)
			for i := range order {
				order[i] = i
			}
			sort.Slice(order, func(a, b int) bool {
				return values[order[a]] > values[order[b]]
			})
			for i := 0; i < numStrike; i++ {
				strikeSet[order[i]] = true
			}
		}

		for i := range individuals {
			if strikeSet[i] {
				for d := range individuals[i] {
					individuals[i][d] = clamp(best[d]+rand.NormFloat64()*energy, lightningSearchRangeLow, lightningSearchRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					direction := 1.0
					if best[d] < individuals[i][d] {
						direction = -1.0
					} else if best[d] == individuals[i][d] {
						direction = 0.0
					}
					step := math.Abs(rand.NormFloat64()*energy) * direction
					individuals[i][d] = clamp(individuals[i][d]+step, lightningSearchRangeLow, lightningSearchRangeHigh)
				}
			}
			values[i] = fn.Evaluate(individuals[i])
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
		Method:     "lightning_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
