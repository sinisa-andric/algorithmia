package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	politicalOptimizerPopulation  = 20
	politicalOptimizerMaxSteps    = 500
	politicalOptimizerNumParties  = 4
	politicalOptimizerTolerance   = 1e-6
	politicalOptimizerRangeLow    = -5.0
	politicalOptimizerRangeHigh   = 5.0
	politicalOptimizerSwitchEvery = 20
)

// PoliticalOptimizer minimizuje konfigurisanu benchmark funkciju koristeći Political Optimizer: članovi stranke
// prate lidera svoje stranke i globalnog lidera, a povremeno najslabiji član nasumične stranke prelazi u trenutno
// najjaču stranku
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func PoliticalOptimizer(problem models.Problem) (result models.Result, err error) {

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

	population := politicalOptimizerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := politicalOptimizerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	numParties := politicalOptimizerNumParties
	if v, ok := problem.Payload["num_parties"].(float64); ok {
		numParties = int(v)
	}
	if numParties < 1 {
		numParties = 1
	}

	tolerance := politicalOptimizerTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	party := make([]int, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = politicalOptimizerRangeLow + rand.Float64()*(politicalOptimizerRangeHigh-politicalOptimizerRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
		party[i] = i % numParties
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

		partyLeaderIdx := make([]int, numParties)
		partyLeaderValue := make([]float64, numParties)
		for p := range partyLeaderValue {
			partyLeaderValue[p] = math.Inf(1)
		}
		for i := range individuals {
			p := party[i]
			if values[i] < partyLeaderValue[p] {
				partyLeaderValue[p] = values[i]
				partyLeaderIdx[p] = i
			}
		}

		for i := range individuals {
			leaderIdx := partyLeaderIdx[party[i]]
			for d := range individuals[i] {
				pullLeader := rand.Float64() * (individuals[leaderIdx][d] - individuals[i][d]) * 0.4
				pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.3
				noise := (rand.Float64()*2 - 1) * 0.15
				individuals[i][d] = clamp(individuals[i][d]+pullLeader+pullBest+noise, politicalOptimizerRangeLow, politicalOptimizerRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		if (steps+1)%politicalOptimizerSwitchEvery == 0 && numParties > 1 {
			randomParty := rand.IntN(numParties)
			worstIdx := -1
			for i := range individuals {
				if party[i] == randomParty && (worstIdx == -1 || values[i] > values[worstIdx]) {
					worstIdx = i
				}
			}
			strongestParty := 0
			for p := 1; p < numParties; p++ {
				if values[partyLeaderIdx[p]] < values[partyLeaderIdx[strongestParty]] {
					strongestParty = p
				}
			}
			if worstIdx != -1 {
				party[worstIdx] = strongestParty
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
		Method:     "political_optimizer",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
