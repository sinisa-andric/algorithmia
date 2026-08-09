package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sghaPopulation      = 20
	sghaMaxSteps        = 500
	sghaMutationRate    = 0.1
	sghaSimplexInterval = 20
	sghaTolerance       = 1e-6
	sghaRangeLow        = -5.0
	sghaRangeHigh       = 5.0
)

// SGHA minimizuje konfigurisanu benchmark funkciju koristeći Simplex-Genetic Hybrid Algorithm: standardna genetska
// evolucija (turnirska selekcija, aritmetičko ukrštanje, mutacija, elitizam) se svakih simplex_interval koraka
// dopunjuje jednim Nelder-Mead refleksija/kontrakcija korakom nad tri najbolje jedinke
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SGHA(problem models.Problem) (result models.Result, err error) {

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

	population := sghaPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sghaMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	mutationRate := sghaMutationRate
	if v, ok := problem.Payload["mutation_rate"].(float64); ok {
		mutationRate = v
	}

	simplexInterval := sghaSimplexInterval
	if v, ok := problem.Payload["simplex_interval"].(float64); ok {
		simplexInterval = int(v)
	}

	tolerance := sghaTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	tournamentSelect3 := func(values []float64) int {
		idx := rand.IntN(population)
		for range 2 {
			c := rand.IntN(population)
			if values[c] < values[idx] {
				idx = c
			}
		}
		return idx
	}

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = sghaRangeLow + rand.Float64()*(sghaRangeHigh-sghaRangeLow)
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

	simplexStep := func() {
		i0, i1, i2 := 0, 1, 2
		if values[i1] < values[i0] {
			i0, i1 = i1, i0
		}
		if values[i2] < values[i1] {
			i1 = i2
			if values[i1] < values[i0] {
				i0, i1 = i1, i0
			}
		}
		for i := 3; i < population; i++ {
			switch {
			case values[i] < values[i0]:
				i0, i1, i2 = i, i0, i1
			case values[i] < values[i1]:
				i1, i2 = i, i1
			case values[i] < values[i2]:
				i2 = i
			}
		}

		centroid := make([]float64, dimensions)
		for d := range centroid {
			centroid[d] = (individuals[i0][d] + individuals[i1][d]) / 2
		}

		reflected := make([]float64, dimensions)
		for d := range reflected {
			reflected[d] = clamp(centroid[d]+(centroid[d]-individuals[i2][d]), sghaRangeLow, sghaRangeHigh)
		}
		reflectedValue := fn.Evaluate(reflected)

		if reflectedValue < values[i2] {
			individuals[i2] = reflected
			values[i2] = reflectedValue
			return
		}

		contracted := make([]float64, dimensions)
		for d := range contracted {
			contracted[d] = clamp(centroid[d]+0.5*(individuals[i2][d]-centroid[d]), sghaRangeLow, sghaRangeHigh)
		}
		contractedValue := fn.Evaluate(contracted)
		if contractedValue < values[i2] {
			individuals[i2] = contracted
			values[i2] = contractedValue
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		children := make([][]float64, population)
		childValues := make([]float64, population)

		children[0] = append([]float64(nil), best...)
		childValues[0] = bestValue

		for i := 1; i < population; i++ {
			parentA := individuals[tournamentSelect3(values)]
			parentB := individuals[tournamentSelect3(values)]

			child := make([]float64, dimensions)
			for d := range child {
				w := rand.Float64()
				child[d] = w*parentA[d] + (1-w)*parentB[d]
			}

			if rand.Float64() < mutationRate {
				d := rand.IntN(dimensions)
				child[d] = child[d] + (rand.Float64()*2-1)*0.3
			}

			for d := range child {
				child[d] = clamp(child[d], sghaRangeLow, sghaRangeHigh)
			}

			children[i] = child
			childValues[i] = fn.Evaluate(child)
		}

		individuals = children
		values = childValues

		if simplexInterval > 0 && steps > 0 && steps%simplexInterval == 0 {
			simplexStep()
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
		Method:     "sgha",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
