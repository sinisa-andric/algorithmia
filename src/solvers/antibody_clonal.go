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
	antibodyClonalPopulation  = 20
	antibodyClonalMaxSteps    = 500
	antibodyClonalCloneFactor = 5
	antibodyClonalCauchyProb  = 0.15
	antibodyClonalTolerance   = 1e-6
	antibodyClonalRangeLow    = -5.0
	antibodyClonalRangeHigh   = 5.0
)

// AntibodyClonal minimizuje konfigurisanu benchmark funkciju koristeći Antibody Clonal Selection: svako antitelo
// se klonira i mutira jačinom obrnuto proporcionalnom rangu afiniteta kao u klonalnoj selekciji, ali povremeno
// umesto Gausove koristi Cauchy mutaciju čiji teški repovi omogućavaju povremene velike skokove van lokalnog minimuma
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func AntibodyClonal(problem models.Problem) (result models.Result, err error) {

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

	population := antibodyClonalPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := antibodyClonalMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	cloneFactor := antibodyClonalCloneFactor
	if v, ok := problem.Payload["clone_factor"].(float64); ok {
		cloneFactor = int(v)
	}

	cauchyProb := antibodyClonalCauchyProb
	if v, ok := problem.Payload["cauchy_prob"].(float64); ok {
		cauchyProb = v
	}

	tolerance := antibodyClonalTolerance
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
			individual[d] = antibodyClonalRangeLow + rand.Float64()*(antibodyClonalRangeHigh-antibodyClonalRangeLow)
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

		for rank, idx := range order {
			mutationScale := 0.05
			if population > 1 {
				mutationScale = 0.05 + (float64(rank)/float64(population-1))*0.95
			}

			bestClone := individuals[idx]
			bestCloneValue := values[idx]
			for range cloneFactor {
				clone := make([]float64, dimensions)
				if rand.Float64() < cauchyProb {
					for d := range clone {
						clone[d] = clamp(individuals[idx][d]+math.Tan((rand.Float64()-0.5)*math.Pi)*0.3, antibodyClonalRangeLow, antibodyClonalRangeHigh)
					}
				} else {
					for d := range clone {
						clone[d] = clamp(individuals[idx][d]+rand.NormFloat64()*mutationScale, antibodyClonalRangeLow, antibodyClonalRangeHigh)
					}
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
		Method:     "antibody_clonal",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
