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
	invasiveWeedPopulation   = 20
	invasiveWeedMaxSteps     = 500
	invasiveWeedMaxSeeds     = 5
	invasiveWeedMinSeeds     = 1
	invasiveWeedSigmaInitial = 1.0
	invasiveWeedSigmaFinal   = 0.01
	invasiveWeedTolerance    = 1e-6
	invasiveWeedRangeLow     = -5.0
	invasiveWeedRangeHigh    = 5.0
)

// InvasiveWeed minimizuje konfigurisanu benchmark funkciju koristeći Invasive Weed Optimization: svaki korov širi
// broj semena proporcionalan sopstvenom fitnesu u odnosu na najgoru jedinku u populaciji, semena se raspršuju
// gausovskim šumom čija disperzija nelinearno opada tokom izvršavanja, a od roditelja i semena zajedno se
// zadržavaju samo najbolje population jedinki
// problem.Point inicijalizuje populaciju korova i određuje njenu dimenzionalnost
func InvasiveWeed(problem models.Problem) (result models.Result, err error) {

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

	population := invasiveWeedPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := invasiveWeedMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	maxSeeds := invasiveWeedMaxSeeds
	if v, ok := problem.Payload["max_seeds"].(float64); ok {
		maxSeeds = int(v)
	}

	minSeeds := invasiveWeedMinSeeds
	if v, ok := problem.Payload["min_seeds"].(float64); ok {
		minSeeds = int(v)
	}

	sigmaInitial := invasiveWeedSigmaInitial
	if v, ok := problem.Payload["sigma_initial"].(float64); ok {
		sigmaInitial = v
	}

	sigmaFinal := invasiveWeedSigmaFinal
	if v, ok := problem.Payload["sigma_final"].(float64); ok {
		sigmaFinal = v
	}

	tolerance := invasiveWeedTolerance
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
			individual[d] = invasiveWeedRangeLow + rand.Float64()*(invasiveWeedRangeHigh-invasiveWeedRangeLow)
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

		progress := float64(maxSteps-steps) / float64(maxSteps)
		sigma := sigmaFinal + (sigmaInitial-sigmaFinal)*progress*progress

		worstValue := values[0]
		for _, v := range values {
			if v > worstValue {
				worstValue = v
			}
		}

		newSeeds := make([][]float64, 0, len(individuals)*maxSeeds)
		newValues := make([]float64, 0, len(individuals)*maxSeeds)
		for i := range individuals {
			numSeeds := minSeeds + int(math.Round(float64(maxSeeds-minSeeds)*(worstValue-values[i])/(worstValue-bestValue+1e-10)))
			if numSeeds < minSeeds {
				numSeeds = minSeeds
			}
			for s := 0; s < numSeeds; s++ {
				seed := make([]float64, dimensions)
				for d := range seed {
					seed[d] = clamp(individuals[i][d]+rand.NormFloat64()*sigma, invasiveWeedRangeLow, invasiveWeedRangeHigh)
				}
				newSeeds = append(newSeeds, seed)
				newValues = append(newValues, fn.Evaluate(seed))
			}
		}

		// spoji roditelje i svu novu semenku, sortiraj po vrednosti i zadrži najboljih population jedinki —
		// veličina privremenog bazena zavisi od broja proizvedenih semena pa selekcija koristi sort/slice
		// umesto fiksnih indeksa, tako da broj poslednjih semena po koraku ne može izazvati grešku van opsega
		pool := make([][]float64, 0, len(individuals)+len(newSeeds))
		pool = append(pool, individuals...)
		pool = append(pool, newSeeds...)
		poolValues := make([]float64, 0, len(values)+len(newValues))
		poolValues = append(poolValues, values...)
		poolValues = append(poolValues, newValues...)

		order := make([]int, len(pool))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return poolValues[order[a]] < poolValues[order[b]]
		})

		keep := population
		if keep > len(pool) {
			keep = len(pool)
		}
		nextIndividuals := make([][]float64, keep)
		nextValues := make([]float64, keep)
		for k := 0; k < keep; k++ {
			nextIndividuals[k] = pool[order[k]]
			nextValues[k] = poolValues[order[k]]
		}
		individuals = nextIndividuals
		values = nextValues

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
		Method:     "invasive_weed",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
