package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	malsPopulation   = 20
	malsMaxSteps     = 500
	malsLocalSteps   = 10
	malsMutationRate = 0.1
	malsTolerance    = 1e-6
	malsRangeLow     = -5.0
	malsRangeHigh    = 5.0
)

// MALS minimizuje konfigurisanu benchmark funkciju koristeći Memetic Algorithm with Local Search: turnirska
// selekcija i aritmetičko ukrštanje proizvode svako dete koje se dorađuje sa nekoliko koraka hill-climbing-a, uz
// elitizam koji čuva najbolju jedinku prethodne generacije
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func MALS(problem models.Problem) (result models.Result, err error) {

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

	population := malsPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := malsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	localSteps := malsLocalSteps
	if v, ok := problem.Payload["local_steps"].(float64); ok {
		localSteps = int(v)
	}

	mutationRate := malsMutationRate
	if v, ok := problem.Payload["mutation_rate"].(float64); ok {
		mutationRate = v
	}

	tolerance := malsTolerance
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

	hillClimb := func(point []float64, value float64) ([]float64, float64) {
		point = append([]float64(nil), point...)
		for s := 0; s < localSteps; s++ {
			d := rand.IntN(dimensions)
			candidate := append([]float64(nil), point...)
			candidate[d] = clamp(candidate[d]+(rand.Float64()*2-1)*0.1, malsRangeLow, malsRangeHigh)
			candidateValue := fn.Evaluate(candidate)
			if candidateValue < value {
				point = candidate
				value = candidateValue
			}
		}
		return point, value
	}

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = malsRangeLow + rand.Float64()*(malsRangeHigh-malsRangeLow)
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

		children := make([][]float64, population)
		childValues := make([]float64, population)

		// elitizam — najbolja jedinka prethodne generacije uvek preživljava
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
				child[d] = clamp(child[d], malsRangeLow, malsRangeHigh)
			}

			childValue := fn.Evaluate(child)
			child, childValue = hillClimb(child, childValue)

			children[i] = child
			childValues[i] = childValue
		}

		individuals = children
		values = childValues

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
		Method:     "mals",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
