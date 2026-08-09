package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	teamGameAlgorithmPopulation = 20
	teamGameAlgorithmMaxSteps   = 500
	teamGameAlgorithmTeamSize   = 4
	teamGameAlgorithmTolerance  = 1e-6
	teamGameAlgorithmRangeLow   = -5.0
	teamGameAlgorithmRangeHigh  = 5.0
)

// TeamGameAlgorithm minimizuje konfigurisanu benchmark funkciju koristeći Team Game Algorithm: jedinke su
// podeljene u fiksne timove, svaka jedinka teži ka najboljoj jedinki sopstvenog tima i ka globalnom best-u, uz
// nezavisan šum
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func TeamGameAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	population := teamGameAlgorithmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := teamGameAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	teamSize := teamGameAlgorithmTeamSize
	if v, ok := problem.Payload["team_size"].(float64); ok {
		teamSize = int(v)
	}
	if teamSize < 1 {
		teamSize = 1
	}

	tolerance := teamGameAlgorithmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	group := make([]int, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = teamGameAlgorithmRangeLow + rand.Float64()*(teamGameAlgorithmRangeHigh-teamGameAlgorithmRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
		group[i] = i / teamSize
	}
	numGroups := (population + teamSize - 1) / teamSize
	if numGroups < 1 {
		numGroups = 1
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

		teamBestIdx := make([]int, numGroups)
		teamBestValue := make([]float64, numGroups)
		for g := range teamBestValue {
			teamBestValue[g] = math.Inf(1)
		}
		for i := range individuals {
			g := group[i]
			if values[i] < teamBestValue[g] {
				teamBestValue[g] = values[i]
				teamBestIdx[g] = i
			}
		}

		for i := range individuals {
			tb := teamBestIdx[group[i]]
			for d := range individuals[i] {
				pullTeam := rand.Float64() * (individuals[tb][d] - individuals[i][d]) * 0.4
				pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.3
				noise := (rand.Float64()*2 - 1) * 0.15
				individuals[i][d] = clamp(individuals[i][d]+pullTeam+pullBest+noise, teamGameAlgorithmRangeLow, teamGameAlgorithmRangeHigh)
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
		Method:     "team_game_algorithm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
