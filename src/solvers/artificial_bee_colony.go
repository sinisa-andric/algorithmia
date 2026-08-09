package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	artificialBeeColonyBees      = 20
	artificialBeeColonyLimit     = 100
	artificialBeeColonyMaxSteps  = 500
	artificialBeeColonyRange     = 15.0
	artificialBeeColonyTolerance = 1e-6
)

// ArtificialBeeColony minimizuje sphere funkciju koristeći algoritam veštačke kolonije pčela: zaposlene pčele
// istražuju oko svakog izvora hrane, pčele posmatrači preko rulet selekcije favorizuju bolje izvore,
// a pčele izviđači zamenjuju izvore koji se nisu poboljšali u okviru limit pokušaja
// problem.Point samo određuje dimenzionalnost
func ArtificialBeeColony(problem models.Problem) (result models.Result, err error) {

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

	bees := artificialBeeColonyBees
	if v, ok := problem.Payload["bees"].(float64); ok {
		bees = int(v)
	}

	limit := artificialBeeColonyLimit
	if v, ok := problem.Payload["limit"].(float64); ok {
		limit = int(v)
	}

	maxSteps := artificialBeeColonyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	searchRange := artificialBeeColonyRange
	if v, ok := problem.Payload["range"].(float64); ok {
		searchRange = v
	}

	tolerance := artificialBeeColonyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	sources := bees / 2
	if sources < 1 {
		sources = 1
	}

	foodSources := make([][]float64, sources)
	counters := make([]int, sources)
	for i := range foodSources {
		solution := make([]float64, dimensions)
		for d := range solution {
			solution[d] = (rand.Float64()*2 - 1) * searchRange
		}
		foodSources[i] = solution
	}

	best := append([]float64(nil), foodSources[0]...)
	bestValue := fn.Evaluate(best)
	for _, solution := range foodSources {
		if value := fn.Evaluate(solution); value < bestValue {
			bestValue = value
			best = append([]float64(nil), solution...)
		}
	}

	explore := func(i int) {
		other := rand.IntN(sources)
		for other == i && sources > 1 {
			other = rand.IntN(sources)
		}

		candidate := make([]float64, dimensions)
		for d := range candidate {
			phi := rand.Float64()*2 - 1
			candidate[d] = foodSources[i][d] + phi*(foodSources[i][d]-foodSources[other][d])
		}

		if value := fn.Evaluate(candidate); value < fn.Evaluate(foodSources[i]) {
			foodSources[i] = candidate
			counters[i] = 0

			if value < bestValue {
				bestValue = value
				best = append([]float64(nil), candidate...)
			}
		} else {
			counters[i]++
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		for i := range foodSources {
			explore(i)
		}

		fitness := make([]float64, sources)
		for i, solution := range foodSources {
			fitness[i] = 1 / (1 + fn.Evaluate(solution))
		}
		for i := 0; i < sources; i++ {
			explore(rouletteSelect(fitness))
		}

		for i := range foodSources {
			if counters[i] > limit {
				solution := make([]float64, dimensions)
				for d := range solution {
					solution[d] = (rand.Float64()*2 - 1) * searchRange
				}
				foodSources[i] = solution
				counters[i] = 0
			}
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "artificial_bee_colony",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func rouletteSelect(fitness []float64) int {

	total := 0.0
	for _, f := range fitness {
		total += f
	}

	target := rand.Float64() * total
	cumulative := 0.0
	for i, f := range fitness {
		cumulative += f
		if target <= cumulative {
			return i
		}
	}

	return len(fitness) - 1
}
