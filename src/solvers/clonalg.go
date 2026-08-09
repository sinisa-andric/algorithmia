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
	clonalgPopulation  = 20
	clonalgMaxSteps    = 500
	clonalgCloneFactor = 5
	clonalgReplaceRate = 0.2
	clonalgTolerance   = 1e-6
	clonalgRangeLow    = -5.0
	clonalgRangeHigh   = 5.0
)

// Clonalg minimizuje konfigurisanu benchmark funkciju koristeći Clonal Selection Algorithm: svako antitelo se
// klonira i hipermutira jačinom obrnuto proporcionalnom rangu afiniteta (najbolje antitelo dobija najfiniju
// pretragu, najgore najširu eksploraciju), a najgori deo populacije se periodično zamenjuje nasumičnim jedinkama
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Clonalg(problem models.Problem) (result models.Result, err error) {

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

	population := clonalgPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := clonalgMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	cloneFactor := clonalgCloneFactor
	if v, ok := problem.Payload["clone_factor"].(float64); ok {
		cloneFactor = int(v)
	}

	replaceRate := clonalgReplaceRate
	if v, ok := problem.Payload["replace_rate"].(float64); ok {
		replaceRate = v
	}

	tolerance := clonalgTolerance
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
			individual[d] = clonalgRangeLow + rand.Float64()*(clonalgRangeHigh-clonalgRangeLow)
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

		// rang po afinitetu (0 = najbolji) određuje jačinu mutacije svakog antitela — bolji rang dobija finu
		// lokalnu mutaciju, gori rang širu eksploraciju; svako antitelo u populaciji prolazi kroz ovu petlju
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
				for d := range clone {
					clone[d] = clamp(individuals[idx][d]+rand.NormFloat64()*mutationScale, clonalgRangeLow, clonalgRangeHigh)
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

		// zamena najgorih replace_rate*population antitela nasumičnim novim radi održavanja diverziteta
		numReplace := int(replaceRate * float64(population))
		if numReplace > 0 {
			worstOrder := make([]int, population)
			for i := range worstOrder {
				worstOrder[i] = i
			}
			sort.Slice(worstOrder, func(a, b int) bool {
				return values[worstOrder[a]] > values[worstOrder[b]]
			})
			for i := 0; i < numReplace && i < population; i++ {
				idx := worstOrder[i]
				individual := make([]float64, dimensions)
				for d := range individual {
					individual[d] = clonalgRangeLow + rand.Float64()*(clonalgRangeHigh-clonalgRangeLow)
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
		Method:     "clonalg",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
