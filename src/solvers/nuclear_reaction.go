package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	nuclearReactionPopulation  = 20
	nuclearReactionMaxSteps    = 500
	nuclearReactionFissionProb = 0.5
	nuclearReactionTolerance   = 1e-6
	nuclearReactionRangeLow    = -5.0
	nuclearReactionRangeHigh   = 5.0
)

// NuclearReaction minimizuje konfigurisanu benchmark funkciju koristeći Nuclear Reaction Optimization: jezgra
// naizmenično prolaze kroz fisiju (nasumično cepanje uz blagu vezu ka best-u) i fuziju (spajanje sa nasumičnim
// peer-om i best-om)
// problem.Point inicijalizuje populaciju jezgara i određuje njenu dimenzionalnost
func NuclearReaction(problem models.Problem) (result models.Result, err error) {

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

	population := nuclearReactionPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := nuclearReactionMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	fissionProb := nuclearReactionFissionProb
	if v, ok := problem.Payload["fission_prob"].(float64); ok {
		fissionProb = v
	}

	tolerance := nuclearReactionTolerance
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
			individual[d] = nuclearReactionRangeLow + rand.Float64()*(nuclearReactionRangeHigh-nuclearReactionRangeLow)
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

		for i := range individuals {
			if rand.Float64() < fissionProb {
				for d := range individuals[i] {
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.2
					noise := (rand.Float64()*2 - 1) * 0.8 * (1 - float64(steps)/float64(maxSteps))
					individuals[i][d] = clamp(individuals[i][d]+noise+pull, nuclearReactionRangeLow, nuclearReactionRangeHigh)
				}
			} else {
				peerIdx := i
				if population > 1 {
					peerIdx = rand.IntN(population)
					for peerIdx == i {
						peerIdx = rand.IntN(population)
					}
				}
				for d := range individuals[i] {
					pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.5
					pullPeer := rand.Float64() * (individuals[peerIdx][d] - individuals[i][d]) * 0.3
					individuals[i][d] = clamp(individuals[i][d]+pullBest+pullPeer, nuclearReactionRangeLow, nuclearReactionRangeHigh)
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
		Method:     "nuclear_reaction",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
